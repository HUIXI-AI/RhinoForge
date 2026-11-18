"""RhinoVLA v3 4/6-core admission, from the example TOML to the cold preflight."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from examples.common import config as common
from examples.common.config import _FIXED_EIGHT as FIXED_EIGHT
from rpu_backend.adapters.rhinovla import pipeline, runtime
from rpu_backend.adapters.rhinovla.convert import convert_expert_for_rpu
from rpu_backend.adapters.rhinovla.fused import RHINO_REDUCED_NATIVE_OFF, _RhinoVLANativeColdConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "examples/configs/rhinovla").rglob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(ROOT)))
def test_example_budgets_follow_the_v3_owner(path):
    # Every v3 recipe, the HIGH family included, has a 4/6-core owner.
    config = common.load_config(path)
    support = common.core_support(config)
    assert support[8] is None
    reason = FIXED_EIGHT["rhinovla"] if "/v1/" in str(path) else None
    assert support[4] == support[6] == reason


def test_v1_profile_refuses_a_reduced_budget():
    config = common.load_config(ROOT / "examples/configs/rhinovla/v1/fp16.toml")
    profile = common.example_profiles()[config["example"]["profile_id"]]
    with pytest.raises(common.ConfigError, match="v1"):
        common._validate_execution({"model": {"num_cores": 6}}, profile=profile)


EIGHT_ONLY = {
    "model": {"num_cores": 8},
    "prefill": {"merger_scatter": True, "prefix_prep_rpu": True, "prefix_prep_cache": True},
    "vision": {"fused_merger": True, "bake_merger": True, "fast_replay": True},
    "components": {"vision_encoder": {"vision": {"fused_merger": True, "chunk_size": 64}}},
}


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_resolver_derives_child_budgets_and_drops_eight_core_selectors(cores):
    request = {**EIGHT_ONLY, "model": {"num_cores": cores}}
    root, text, vision, action = runtime.resolve_rhinovla_execution(request, entry_point="test")
    dropped = {stage: set(root.get(stage, {})) & set(fields)
               for stage, fields in runtime.RHINOVLA_REDUCED_OFF.items()}
    assert dropped == {"prefill": set(), "vision": set()}
    if cores == 8:
        assert all("model" not in child for child in (text, vision, action))
        return
    assert root["prefill"]["prefix_prep_rpu"] and root["vision"]["fast_replay"]
    assert root["prefill"]["merger_scatter"]
    assert root["vision"]["fused_merger"] and root["vision"]["bake_merger"]
    assert root["components"]["vision_encoder"]["vision"] == {"fused_merger": True, "chunk_size": 64}
    assert [child["model"]["num_cores"] for child in (text, vision, action)] == [cores, 4, cores]
    snapshot, _ = runtime.RhinoVLAPipelineColdConfig.resolve(request)
    assert snapshot.stages["prefill"]["merger_scatter"]
    assert snapshot.stages["vision"]["fused_merger"]
    assert snapshot.stages["vision"]["bake_merger"]


def test_reduced_budget_overrides_eight_core_env_selectors(monkeypatch):
    monkeypatch.setenv("RPU_QWEN3VL_VISION_FUSED_MERGER", "1")
    monkeypatch.setenv("RPU_RHINOVLA_VISION_BAKE_MERGER", "1")
    monkeypatch.setenv("RPU_RHINOVLA_VISION_FAST_REPLAY", "1")
    snapshot, _ = runtime.RhinoVLAPipelineColdConfig.resolve({"model": {"num_cores": 6}})
    assert snapshot.stages["vision"]["fused_merger"]
    assert snapshot.stages["vision"]["bake_merger"]
    assert snapshot.stages["vision"]["fast_replay"]


@pytest.mark.parametrize("cores", (4, 6))
def test_factory_entry_forwards_the_composite_budget(cores):
    from rpu_backend.api.rhinovla import RhinoVLAPolicy

    class Built(Exception):
        pass

    def factory(config, *, rpu_execution):
        raise Built(rpu_execution)

    with pytest.raises(Built) as built:
        RhinoVLAPolicy.from_factory(factory, {}, rpu_execution={"model": {"num_cores": cores}})
    assert built.value.args[0]["model"] == {"num_cores": cores}


def test_component_cannot_choose_its_own_budget():
    with pytest.raises(ValueError):
        runtime.resolve_rhinovla_execution(
            {"components": {"action_expert": {"model": {"num_cores": 4}}}}, entry_point="test")


def test_native_selectors_without_reduced_owner_turn_off(monkeypatch):
    for name in _RhinoVLANativeColdConfig.__dataclass_fields__:
        monkeypatch.setenv("RPU_RHINOVLA_" + name.upper(), "0")
    for name in (*RHINO_REDUCED_NATIVE_OFF, "precompute_adarms", "gated_no_sub"):
        monkeypatch.setenv("RPU_RHINOVLA_" + name.upper(), "1")
    eight = _RhinoVLANativeColdConfig.from_env()
    assert all(getattr(eight, name) for name in RHINO_REDUCED_NATIVE_OFF)
    reduced = _RhinoVLANativeColdConfig.from_env(num_cores=6)
    assert not any(getattr(reduced, name) for name in RHINO_REDUCED_NATIVE_OFF)
    assert reduced.precompute_adarms and reduced.gated_no_sub


@pytest.mark.parametrize("cores", (5, True, 2))
def test_expert_conversion_rejects_a_budget_without_owner(cores):
    expert = torch.nn.Module()
    with pytest.raises(ValueError, match="4, 6 or 8"):
        convert_expert_for_rpu(expert, num_cores=cores)
    assert not getattr(expert, "_rpu_swizzle_started", False)


def test_controller_keeps_the_built_budget():
    controller = NS(action_expert=NS(_rpu_execution_topology=(6, 4, 6, 8)))
    with pytest.raises(ValueError, match="cold-bound"):
        runtime._RhinoVLAExecutionController.validate(controller, {"model": {"num_cores": 8}})


def _text_config():
    return NS(model_type="qwen3_vl_text", hidden_size=2048, intermediate_size=6144,
              num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
              head_dim=128, vocab_size=151936, hidden_act="silu", rms_norm_eps=1e-6,
              attention_bias=False, tie_word_embeddings=True, use_cache=True,
              rope_theta=5_000_000.0,
              rope_parameters={"rope_type": "default", "rope_theta": 5_000_000.0,
                               "mrope_interleaved": True, "mrope_section": [24, 20, 20]})


HIGH_FAMILY = ("HIGH_PRECISION", "HIGH_PRECISION_FUSIONS", "VECTOR_K_NORM", "VECTOR_Q_NORM",
               "PARTIAL_ROPE", "PRECOMPUTE_GATE_TANH")


@pytest.mark.parametrize("fault", (
    None, "eight", "high", "gate_tanh", "eight_high", "eight_vector_k", "unroll", "alias",
    "adarms", "streaming", "geometry", "action_io", "vision", "text", "limit"))
def test_cold_preflight_admits_only_the_v3_reduced_owners(monkeypatch, fault):
    from rpu_backend.runtime import hw_attrs

    for name in _RhinoVLANativeColdConfig.__dataclass_fields__:
        monkeypatch.delenv("RPU_RHINOVLA_" + name.upper(), raising=False)
    for name in (*HIGH_FAMILY, "EXPERT_DEPTH_LIMIT", "EXPERT_MLP_DIM_LIMIT"):
        monkeypatch.delenv("RPU_RHINOVLA_" + name, raising=False)
    monkeypatch.setenv("RPU_RHINOVLA_FULL_W8A16", "1")
    monkeypatch.setenv("RPU_RHINOVLA_PRECOMPUTE_ADARMS", "0" if fault == "adarms" else "1")
    eight = fault in ("eight", "eight_high", "eight_vector_k")
    if not eight:
        # An eight-core TOML may name these; the reduced budget turns them off.
        for name in RHINO_REDUCED_NATIVE_OFF:
            monkeypatch.setenv("RPU_RHINOVLA_" + name.upper(), "1")
    monkeypatch.setenv("RPU_RHINOVLA_GATED_NO_SUB", "1")
    # Reduced budgets run the whole HIGH family without the resident tables
    # and aligned KV; eight cores still require both.
    high = {"high": HIGH_FAMILY, "gate_tanh": ("PRECOMPUTE_GATE_TANH",),
            "eight_high": ("HIGH_PRECISION",),
            "eight_vector_k": ("HIGH_PRECISION", "VECTOR_K_NORM")}.get(fault, ())
    for name in high:
        monkeypatch.setenv("RPU_RHINOVLA_" + name, "1")
    if fault == "limit":
        monkeypatch.setenv("RPU_RHINOVLA_EXPERT_DEPTH_LIMIT", "9")
    monkeypatch.setattr(hw_attrs, "validate_preinstall", lambda _: None)
    tc = _text_config()
    vc = NS(depth=24, hidden_size=1024, intermediate_size=4096, num_heads=16)
    if fault == "text":
        tc.rms_norm_eps = 1e-5
    if fault == "vision":
        vc.depth = 23
    model = NS(qwen=NS(model=NS(model=NS(language_model=object(), visual=object()),
                               config=NS(text_config=tc, vision_config=vc))))
    cfg = NS(depth=18, mlp_dim=2048 if fault == "geometry" else 3072, action_horizon=30,
             use_state_token=True, width=1024, num_attention_heads=16,
             num_key_value_heads=8, head_dim=128)
    action_io = NS(action_time_mlp_in=object(), action_time_mlp_out=object()) \
        if fault == "action_io" else NS()
    execution = {
        "model": {"num_cores": 8 if eight else 6},
        "action": {"denoise_unroll": fault != "unroll", "shared_cache": True,
                   "prefix_alias": fault != "alias"},
        "vision": {"rpu_patch_embed": True, "rpu_mergers": True, "fused_merger": True,
                   "rpu_mergers_streaming": fault == "streaming"},
    }

    def preflight():
        return pipeline._preflight_rhinovla_cold_install(
            qin={}, prefix_len=230, steps=10, rpu_execution=execution, model=model,
            action_bundle=(object(), action_io, object(), cfg),
            # The official flow already forbids the perf-only depth limit.
            flow_direction="legacy_ascending" if fault == "limit" else "official_descending")

    errors = {
        "eight_high": "resident AdaRMS tables", "eight_vector_k": "ALIGNED_KV",
        "unroll": "requires action.denoise_unroll", "alias": "requires action.denoise_unroll",
        "adarms": "requires PRECOMPUTE_ADARMS", "streaming": "no streaming-merger owner",
        "geometry": "v3 18-layer", "action_io": "v3 18-layer", "vision": "v3 24-layer",
        "text": "exact Qwen3-VL-2B", "limit": "no depth/MLP-limited owner",
    }
    if fault in errors:
        with pytest.raises(ValueError, match=errors[fault]):
            preflight()
        return
    cold = preflight()
    if fault == "eight":
        assert cold["cores"] == 8 and cold["text_topology"] is None
        return
    topology = cold["text_topology"]
    assert cold["cores"] == 6 and cold["full_w8a16"]
    assert (topology.num_cores, topology.attn_tp, topology.mlp_tp) == (6, 4, 6)
    for name in HIGH_FAMILY:
        assert cold[name.lower()] is (name in high)
