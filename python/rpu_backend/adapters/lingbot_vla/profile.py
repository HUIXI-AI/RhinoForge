"""Cold, board-free contract for the certified LingBot-VLA-4B checkpoint."""
from __future__ import annotations

from rpu_backend.runtime import rpu_env_bool

# Checkpoint key prefixes.
VLM_P = "model.qwenvl_with_expert.qwenvl.model."
EXP_P = "model.qwenvl_with_expert.qwen_expert.model."
VIS_P = "model.qwenvl_with_expert.qwenvl.visual."

# VLM decoder / expert / shared geometry.
VLM_LAYERS = 36
VLM_HID = 2048
VLM_INTER = 11008
VLM_VOCAB = 151936
EXP_LAYERS = 36
EXP_HID = 768
EXP_INTER = 2752
EXP_INTER_PAD = 2816
NQ, NKV, HD = 16, 2, 128
RMS_EPS = 1e-6
ROPE_BASE = 10_000.0
N_ACTION = 50
NUM_STEPS = 10
ACTION_DIM = 75
ATTN_TP = 2
MLP_CORES = 8
DEFAULT_MAX_SEQ = 256

# Qwen2.5-VL Vision geometry.
V_DEPTH = 32
V_HID = 1280
V_HEADS = 16
V_HD = V_HID // V_HEADS
V_PATCH = 14
V_SM = 2
V_SMU = V_SM * V_SM
V_WINDOW = 112
V_FULLATT = {7, 15, 23, 31}
V_OUT = 2048
V_THETA = 10_000.0
V_INTER = 3420

def _runtime_options_from_env() -> dict[str, bool]:
    """Read the six runtime switches, one literal env name per read.

    The names are spelled out at the call site on purpose. They used to live in
    a `{option: (env_name, default)}` table consumed by a comprehension, and
    `scripts/check_runtime_config_env.py` walks the AST for constant first
    arguments — a name reached through a loop variable is invisible to it, so
    renaming one here was unenforced by the docs contract. Do not re-table them.
    """
    return {
        "expert_replay": rpu_env_bool("RPU_LINGBOT_EXPERT_REPLAY", default=True),
        "vlm_replay": rpu_env_bool("RPU_LINGBOT_VLM_REPLAY", default=True),
        "fused_encoder": rpu_env_bool("RPU_LINGBOT_FUSED_ENCODER", default=True),
        "encoder_1thread": rpu_env_bool(
            "RPU_LINGBOT_ENCODER_1THREAD", default=True),
        "ondevice_encoder": rpu_env_bool("RPU_LINGBOT_ONDEVICE_ENCODER"),
        "denoise_unroll": rpu_env_bool("RPU_LINGBOT_DENOISE_UNROLL"),
    }


def _expected_checkpoint_shapes() -> dict[str, tuple[int, ...]]:
    """Exact tensor geometry consumed by the certified LingBot-VLA-4B path."""
    expected = {
        VLM_P + "embed_tokens.weight": (VLM_VOCAB, VLM_HID),
        VLM_P + "norm.weight": (VLM_HID,),
        EXP_P + "norm.weight": (EXP_HID,),
        VIS_P + "patch_embed.proj.weight": (V_HID, 3, 2, V_PATCH, V_PATCH),
        VIS_P + "merger.ln_q.weight": (V_HID,),
        VIS_P + "merger.mlp.0.weight": (V_HID * V_SMU, V_HID * V_SMU),
        VIS_P + "merger.mlp.0.bias": (V_HID * V_SMU,),
        VIS_P + "merger.mlp.2.weight": (V_OUT, V_HID * V_SMU),
        VIS_P + "merger.mlp.2.bias": (V_OUT,),
        "model.state_proj.weight": (EXP_HID, ACTION_DIM),
        "model.state_proj.bias": (EXP_HID,),
        "model.action_in_proj.weight": (EXP_HID, ACTION_DIM),
        "model.action_in_proj.bias": (EXP_HID,),
        "model.action_out_proj.weight": (ACTION_DIM, EXP_HID),
        "model.action_out_proj.bias": (ACTION_DIM,),
        "model.action_time_mlp_in.weight": (EXP_HID, EXP_HID * 2),
        "model.action_time_mlp_in.bias": (EXP_HID,),
        "model.action_time_mlp_out.weight": (EXP_HID, EXP_HID),
        "model.action_time_mlp_out.bias": (EXP_HID,),
    }
    q_dim = NQ * HD
    kv_dim = NKV * HD
    for layer in range(VLM_LAYERS):
        prefix = f"{VLM_P}layers.{layer}."
        expected.update({
            prefix + "self_attn.q_proj.weight": (q_dim, VLM_HID),
            prefix + "self_attn.q_proj.bias": (q_dim,),
            prefix + "self_attn.k_proj.weight": (kv_dim, VLM_HID),
            prefix + "self_attn.k_proj.bias": (kv_dim,),
            prefix + "self_attn.v_proj.weight": (kv_dim, VLM_HID),
            prefix + "self_attn.v_proj.bias": (kv_dim,),
            prefix + "self_attn.o_proj.weight": (VLM_HID, q_dim),
            prefix + "mlp.gate_proj.weight": (VLM_INTER, VLM_HID),
            prefix + "mlp.up_proj.weight": (VLM_INTER, VLM_HID),
            prefix + "mlp.down_proj.weight": (VLM_HID, VLM_INTER),
            prefix + "input_layernorm.weight": (VLM_HID,),
            prefix + "post_attention_layernorm.weight": (VLM_HID,),
        })
    for layer in range(EXP_LAYERS):
        prefix = f"{EXP_P}layers.{layer}."
        expected.update({
            prefix + "self_attn.q_proj.weight": (q_dim, EXP_HID),
            prefix + "self_attn.q_proj.bias": (q_dim,),
            prefix + "self_attn.k_proj.weight": (kv_dim, EXP_HID),
            prefix + "self_attn.k_proj.bias": (kv_dim,),
            prefix + "self_attn.v_proj.weight": (kv_dim, EXP_HID),
            prefix + "self_attn.v_proj.bias": (kv_dim,),
            prefix + "self_attn.o_proj.weight": (EXP_HID, q_dim),
            prefix + "mlp.gate_proj.weight": (EXP_INTER, EXP_HID),
            prefix + "mlp.up_proj.weight": (EXP_INTER, EXP_HID),
            prefix + "mlp.down_proj.weight": (EXP_HID, EXP_INTER),
        })
        for slot in ("input_layernorm", "post_attention_layernorm"):
            norm = prefix + slot
            expected.update({
                norm + ".weight": (EXP_HID,),
                norm + ".gamma.weight": (EXP_HID, EXP_HID),
                norm + ".gamma.bias": (EXP_HID,),
                norm + ".beta.weight": (EXP_HID, EXP_HID),
                norm + ".beta.bias": (EXP_HID,),
            })
    for layer in range(V_DEPTH):
        prefix = f"{VIS_P}blocks.{layer}."
        expected.update({
            prefix + "attn.qkv.weight": (V_HID * 3, V_HID),
            prefix + "attn.qkv.bias": (V_HID * 3,),
            prefix + "attn.proj.weight": (V_HID, V_HID),
            prefix + "attn.proj.bias": (V_HID,),
            prefix + "norm1.weight": (V_HID,),
            prefix + "norm2.weight": (V_HID,),
            prefix + "mlp.gate_proj.weight": (V_INTER, V_HID),
            prefix + "mlp.gate_proj.bias": (V_INTER,),
            prefix + "mlp.up_proj.weight": (V_INTER, V_HID),
            prefix + "mlp.up_proj.bias": (V_INTER,),
            prefix + "mlp.down_proj.weight": (V_HID, V_INTER),
            prefix + "mlp.down_proj.bias": (V_HID,),
        })
    return expected


def _validate_checkpoint_profile(key_to_reader) -> None:
    """Reject missing/mismatched/non-fp32 weights before RPU materialization."""
    errors = []
    for prefix, expected_depth in (
        (VLM_P + "layers.", VLM_LAYERS),
        (EXP_P + "layers.", EXP_LAYERS),
        (VIS_P + "blocks.", V_DEPTH),
    ):
        actual_layers = {
            int(remainder.split(".", 1)[0])
            for key in key_to_reader
            if key.startswith(prefix)
            for remainder in (key[len(prefix):],)
            if remainder.split(".", 1)[0].isdigit()
        }
        required_layers = set(range(expected_depth))
        if actual_layers != required_layers:
            errors.append(
                f"{prefix} layer indices {sorted(actual_layers)}, "
                f"expected 0..{expected_depth - 1}"
            )
    for key, expected_shape in _expected_checkpoint_shapes().items():
        if len(errors) >= 8:
            break
        reader = key_to_reader.get(key)
        if reader is None:
            errors.append(f"missing {key}")
        else:
            tensor_slice = reader.get_slice(key)
            actual_shape = tuple(tensor_slice.get_shape())
            if actual_shape != expected_shape:
                errors.append(
                    f"{key}: shape {actual_shape}, expected {expected_shape}"
                )
            if str(tensor_slice.get_dtype()) != "F32":
                errors.append(
                    f"{key}: dtype {tensor_slice.get_dtype()}, expected F32"
                )
    if errors:
        detail = "; ".join(errors)
        raise ValueError(
            "LingBot-VLA checkpoint does not match the certified 4B fp32 "
            f"profile: {detail}."
        )
