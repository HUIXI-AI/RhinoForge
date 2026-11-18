"""Official GPTQ admission and its real public execution boundaries."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_5_moe.quant_scope import (
    EXACT_PROFILE_ID, GPTQ_METHOD, GPTQ_PROFILE_ID, RTN_PROFILE_ID,
    runtime_capabilities, validate_exact_profile,
)
from rpu_backend.api.errors import UnsupportedModelError
from test_qwen35_moe_python_integration import _exact_config


def official_config():
    # The quantization declaration in Qwen's released 7baa396c checkpoint.
    cfg = _exact_config()
    cfg.pop("quant_config")
    cfg["quantization_config"] = {
        "bits": 4, "group_size": 128, "damp_percent": 0.01,
        "desc_act": False, "static_groups": False, "sym": True,
        "true_sequential": True, "quant_method": "gptq",
        "dynamic": {"lm_head": {}, "model.language_model.embed_tokens": {},
                    "-:.*attn.*": {}, "-:.*shared_expert.*": {},
                    "-:.*mtp.*": {}, "-:.*visual.*": {}},
        "modules_to_not_convert": [],
    }
    return cfg


def test_stock_hf_config_selects_gptq_without_rewriting_its_declaration():
    from transformers import Qwen3_5MoeConfig
    from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
    from rpu_backend.adapters.qwen3_5_moe.loader import parse_quant_config

    raw = official_config()
    config = Qwen3_5MoeConfig.from_dict(raw)
    quant = parse_quant_config(config)
    assert quant["profile"] == GPTQ_PROFILE_ID and quant["method"] == GPTQ_METHOD
    assert quant["qzeros_encoding"] == "direct_zero" and quant["effective_zero"] == 8
    assert config.quantization_config == raw["quantization_config"]
    assert not hasattr(config, "rtn_metadata")
    Qwen3_5MoeAdapter.preflight(config)
    Qwen3_5MoeAdapter.preflight_execution(config, {"model": {"num_cores": 8}})


@pytest.mark.parametrize("field,value", [
    ("bits", True), ("bits", 8), ("group_size", 64), ("sym", False),
    ("desc_act", True), ("static_groups", True), ("quant_method", "awq"),
    ("checkpoint_format", "gptq_v2"), ("checkpoint_format", None),
    ("dynamic", {}), ("modules_to_not_convert", ["lm_head"]),
])
def test_unsupported_official_semantics_fail_closed(field, value):
    cfg = official_config()
    cfg["quantization_config"][field] = value
    with pytest.raises(ValueError, match="GPTQ"):
        validate_exact_profile(cfg)


@pytest.mark.parametrize("metadata", [None, {}, {"algorithm": "GPTQ"},
    {"algorithm": "RTN", "manifest": "derivation_manifest.json", "calibration": 0}])
def test_damaged_rtn_marker_cannot_fall_back_to_official_gptq(metadata):
    cfg = official_config()
    cfg["rtn_metadata"] = metadata
    with pytest.raises(ValueError, match="RTN.*derivation metadata"):
        validate_exact_profile(cfg)


@pytest.mark.parametrize("extra", [None, _exact_config()["quant_config"]])
def test_gptq_and_mixed_fp8_declarations_conflict(extra):
    cfg = official_config()
    cfg["quant_config"] = extra
    with pytest.raises(ValueError, match="combine.*FP8.*GPTQ"):
        validate_exact_profile(cfg)


def test_exact_rtn_marker_keeps_its_source_identity():
    cfg = official_config()
    cfg["rtn_metadata"] = {"algorithm": "RTN", "manifest": "derivation_manifest.json",
                           "calibration": False}
    assert validate_exact_profile(cfg)["profile"] == RTN_PROFILE_ID
    assert runtime_capabilities(RTN_PROFILE_ID) == runtime_capabilities(GPTQ_PROFILE_ID)
    assert runtime_capabilities(EXACT_PROFILE_ID).mixer_precision == "fp8_e4m3"
    with pytest.raises(ValueError, match="Unknown"):
        runtime_capabilities("arbitrary_mode6")


@pytest.mark.parametrize("cores", [4, 6])
def test_gptq_reduced_tp_preflight_and_cache_share_topology(cores):
    from transformers import Qwen3_5MoeConfig
    from torch._subclasses.fake_tensor import FakeTensorMode
    from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache

    cfg = Qwen3_5MoeConfig.from_dict(official_config())
    execution = {"model": {"num_cores": cores}}
    Qwen3_5MoeAdapter.preflight_execution(cfg, execution)
    with FakeTensorMode():
        cache = Qwen3_5MoeCache.from_config(cfg, 64, device="cpu", num_cores=cores)
    assert cache.execution_topology == (cores, 4, cores, cores, 8)


@pytest.mark.parametrize("profile", [RTN_PROFILE_ID, GPTQ_PROFILE_ID])
@pytest.mark.parametrize("field", ["pixel_values_videos", "video_grid_thw"])
def test_int4_video_input_rejected_before_forward_mutation(monkeypatch, profile, field):
    from rpu_backend.adapters import qwen3_5_moe as adapter
    from rpu_backend.runtime import hw_attrs

    monkeypatch.setattr(hw_attrs, "mark_first_forward_done",
                        lambda *_: pytest.fail("video input reached forward mutation"))
    model = SimpleNamespace(_qwen3_5_moe_profile=profile)
    with pytest.raises(NotImplementedError, match="video inputs"):
        adapter._rpu_qwen3_5_moe_forward(model, **{field: torch.zeros(1)})


@pytest.mark.parametrize("profile", [RTN_PROFILE_ID, GPTQ_PROFILE_ID])
@pytest.mark.parametrize("field", ["input_ids", "pixel_values", "image_grid_thw",
                                  "mm_token_type_ids"])
def test_int4_incomplete_image_request_rejected_before_mutation(monkeypatch, profile, field):
    from rpu_backend.adapters import qwen3_5_moe as adapter
    from rpu_backend.runtime import hw_attrs

    monkeypatch.setattr(hw_attrs, "mark_first_forward_done",
                        lambda *_: pytest.fail("incomplete image request mutated state"))
    inputs = dict(input_ids=torch.zeros((1, 6), dtype=torch.long),
                  pixel_values=torch.zeros((16, 1536)),
                  image_grid_thw=torch.tensor([[1, 4, 4]]),
                  mm_token_type_ids=torch.tensor([[0, 1, 1, 1, 1, 0]]))
    inputs.pop(field)
    with pytest.raises(ValueError, match="image prefill requires"):
        adapter._rpu_qwen3_5_moe_forward(
            SimpleNamespace(_qwen3_5_moe_profile=profile), **inputs)


@pytest.mark.parametrize("profile", [RTN_PROFILE_ID, GPTQ_PROFILE_ID])
def test_int4_image_prefill_reaches_shared_multimodal_planner(monkeypatch, profile):
    from rpu_backend.adapters import qwen3_5_moe as adapter
    from rpu_backend.runtime import hw_attrs

    class ReachedPlanner(Exception):
        pass

    class Cache:
        position = 0

    cache = Cache()
    language = SimpleNamespace()
    model = SimpleNamespace(_qwen3_5_moe_profile=profile,
                            model=SimpleNamespace(language_model=language, visual=object()))
    monkeypatch.setattr(adapter, "Qwen3_5MoeCache", Cache)
    monkeypatch.setattr(hw_attrs, "mark_first_forward_done", lambda *_: None)

    def preflight(inner, actual_cache, length, *, multimodal_prefill):
        assert inner is language and actual_cache is cache
        assert length == 6 and multimodal_prefill is True
        raise ReachedPlanner

    monkeypatch.setattr(adapter, "_preflight_qwen3_5_moe_text", preflight)
    with pytest.raises(ReachedPlanner):
        adapter._rpu_qwen3_5_moe_forward(
            model, input_ids=torch.zeros((1, 6), dtype=torch.long),
            past_key_values=cache, pixel_values=torch.zeros((16, 1536)),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            mm_token_type_ids=torch.tensor([[0, 1, 1, 1, 1, 0]]))


@pytest.mark.parametrize("profile", [EXACT_PROFILE_ID, RTN_PROFILE_ID, GPTQ_PROFILE_ID])
@pytest.mark.parametrize("field,shape", [("input_ids", (2, 4)),
                                       ("inputs_embeds", (2, 4, 2048))])
def test_batch_two_rejected_before_cache_planning_or_forward_mutation(monkeypatch, profile, field, shape):
    from rpu_backend.adapters import qwen3_5_moe as adapter
    from rpu_backend.runtime import hw_attrs

    monkeypatch.setattr(hw_attrs, "mark_first_forward_done",
                        lambda *_: pytest.fail("unsupported batch mutated forward state"))
    monkeypatch.setattr(adapter, "_preflight_qwen3_5_moe_text",
                        lambda *a, **k: pytest.fail("unsupported batch reached the planner"))
    model = SimpleNamespace(_qwen3_5_moe_profile=profile)
    with pytest.raises(UnsupportedModelError, match="batch_size == 1"):
        adapter._rpu_qwen3_5_moe_forward(model, **{field: torch.zeros(shape)})


@pytest.mark.parametrize("field,shape", [("input_ids", (4,)),
                                       ("inputs_embeds", (4, 2048))])
def test_missing_batch_axis_rejected_before_forward_mutation(monkeypatch, field, shape):
    from rpu_backend.adapters import qwen3_5_moe as adapter
    from rpu_backend.runtime import hw_attrs

    monkeypatch.setattr(hw_attrs, "mark_first_forward_done",
                        lambda *_: pytest.fail("malformed input mutated forward state"))
    model = SimpleNamespace(_qwen3_5_moe_profile=GPTQ_PROFILE_ID)
    with pytest.raises(ValueError, match="input_ids.*batch, seq"):
        adapter._rpu_qwen3_5_moe_forward(model, **{field: torch.zeros(shape)})


def test_resident_preflight_decouples_outer_window_from_sort_capacity():
    from transformers import Qwen3_5MoeConfig
    from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
    from rpu_backend.adapters.qwen3_5_moe.text import _qwen3_5_moe_prefill_envelope
    cfg = Qwen3_5MoeConfig.from_dict(official_config())
    execution = {"prefill": {"chunk_size": 768, "moe_resident_prefill": True}}
    Qwen3_5MoeAdapter.preflight_execution(cfg, execution)
    assert _qwen3_5_moe_prefill_envelope(40, 2048, 8192, resident_prefill=True) == (8320, 8320)
    execution["prefill"].update(moe_resident_prefill=False, chunk_size=768)
    with pytest.raises(UnsupportedModelError, match="exceeds"):
        Qwen3_5MoeAdapter.preflight_execution(cfg, execution)
    # Native joint planning proves the actual SPM and SDPA schedule. Python
    # must not reintroduce a 3x256 router-derived outer-window ceiling.
    for rows in (832, 1024, 1536):
        execution["prefill"].update(moe_resident_prefill=True, chunk_size=rows)
        Qwen3_5MoeAdapter.preflight_execution(cfg, execution)
    mixed = Qwen3_5MoeConfig.from_dict(_exact_config())
    Qwen3_5MoeAdapter.preflight_execution(mixed, {"prefill": {"moe_resident_prefill": True}})


def test_route_table_capacity_is_u16_domain_not_outer_512_cap():
    from rpu_backend.adapters.qwen3_5_moe.weights import build_route_tables
    expert, token = build_route_tables(768, 256, 8, device="cpu")
    assert expert.shape == (768, 256) and token.shape == (768, 8)
    assert torch.equal(token[:, 0], torch.arange(768, dtype=torch.int16))
    with pytest.raises(ValueError, match="65536"):
        build_route_tables(65537, 256, 8, device="cpu")
