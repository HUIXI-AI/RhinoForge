"""Board-free contracts for Qwen3-VL component planning."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_vl import vision
from rpu_backend.adapters.qwen3_vl import Qwen3VLAdapter
from rpu_backend.api import conditional_generation
from rpu_backend.runtime.execution_planner import (
    GRAPH_COMPOSITE_CHILD,
    StageTuple,
)


ROOT = Path(__file__).resolve().parents[1]


def test_vision_dry_plan_preserves_the_native_descriptor(monkeypatch):
    calls = []
    descriptor = (9, 8, 7, 6)

    def resolve(handle, length, image_count):
        calls.append((handle, length, image_count))
        return (
            StageTuple(
                256,
                144,
                144,
                physical_descriptor=descriptor,
            ),
        )

    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3vl_vision_resolve_stage_domain",
        resolve,
        raising=False,
    )
    result = vision._plan_qwen3_vl_vision_execution(11, 256, 1, 144)

    assert calls == [(11, 256, 1)]
    assert result.graph_mode == GRAPH_COMPOSITE_CHILD
    assert result.selected is not None
    assert result.selected.stage_tuple.physical_descriptor == descriptor
    assert result.selected.stage_tuple.compute_chunk == 144


@pytest.mark.parametrize("value", [[1], (True,), (-1,), ("1",)])
def test_vision_parent_graph_key_hook_fails_closed(value):
    with pytest.raises(RuntimeError, match="tuple of non-negative integers"):
        vision._execution_graph_key_words(
            SimpleNamespace(_rpu_execution_graph_key_words=value)
        )


def test_vision_parent_graph_key_hook_defaults_to_empty():
    assert vision._execution_graph_key_words(SimpleNamespace()) == ()
    assert vision._execution_graph_key_words(
        SimpleNamespace(_rpu_execution_graph_key_words=(0, 17))
    ) == (0, 17)


def test_native_vision_dry_descriptor_is_consumed_as_sole_stage_authority():
    model = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    dispatch = (
        ROOT / "src/core/rpu_dispatch_registrations.inc"
    ).read_text()

    assert "rpu_qwen3vl_vision_resolve_stage_domain" in model
    assert "resolve_prefill_stage_domain_for_shape(" in model
    assert "encode_fmb_prefill_stage_domain(" in model
    assert "set_chunk_size_override(0);" in model
    assert "subclass_chunk_size_cap(" in model
    assert "subclass_chunk_size_valid(" in model
    assert "planned_stage_descriptor.empty()" in model
    assert "/*planned_chunk_size=*/0," in model
    assert "planned_stage_descriptor);" in model
    assert "rpu_qwen3vl_vision_resolve_stage_domain" in decls
    assert "rpu_qwen3vl_vision_set_rope_route" in decls
    assert "int[] planned_stage_descriptor=[]" in dispatch
    assert "TORCH_FN(rpu_qwen3vl_vision_resolve_stage_domain)" in dispatch
    assert "TORCH_FN(rpu_qwen3vl_vision_set_rope_route)" in dispatch


def test_vision_rope_domain_and_every_execution_path_share_descriptor_authority():
    model = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    header = (ROOT / "src/core/fused_model_base.h").read_text()
    rhino = (ROOT / "python/rpu_backend/adapters/rhinovla/vision.py").read_text()

    assert "physical_manifest_domain_for_candidate(" in header
    assert "physical_manifest_domain_for_candidate(" in base
    assert "physical_manifest_domain_for_candidate(" in model
    assert "/*rope_spm=*/false" in model
    assert "/*rope_spm=*/true" in model
    assert model.count("bind_descriptor_layout(") >= 4
    assert "selected_rope_residency(" in model
    assert "FMB_ROUTE_FLAG_ROPE_TABLE_DDR" in model
    assert "FMB_ROUTE_FLAG_ROPE_TABLE_SPM" in model
    assert "layout.rope_table_residency" in model
    assert (
        "? rope_route_request_ == Qwen3VLVisionRopeRouteRequest::SPM"
        in model
    )
    assert "qwen3vl_vision_set_rope_route(" in rhino

    # Prepared candidates reuse their already verified physical manifest;
    # the fallback binds the same manifest before buffer declaration.
    residency = base.index(
        "layout_ctx.rope_table_residency = prepared_candidate->rope_residency();"
    )
    attention = base.index("layout_ctx.attention_policy = *attention;", residency)
    fallback = base.index(
        "layout_ctx = bind_physical_layout_context(layout_ctx, physical_manifest);",
        attention,
    )
    decl = base.index("auto decl_fn =", fallback)
    exact_probe = base.index("LayoutContext exact_probe_ctx = layout_ctx;", decl)
    exact_decls = base.index("decl_fn(exact_probe_ctx);", exact_probe)
    assert residency < attention < fallback < decl < exact_probe < exact_decls


@pytest.mark.parametrize(
    ("value", "expected"),
    ((None, 0), ("1", 1), ("0", 2)),
)
def test_vision_rope_legacy_env_is_a_cold_exact_translator(
    monkeypatch, value, expected,
):
    name = "RPU_QWEN3VL_VISION_ROPE_SPM"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert vision._qwen3vl_rope_route_request() == expected


def test_conditional_loader_accepts_registered_component_overrides(monkeypatch):
    class Model:
        def eval(self):
            return self

    class Adapter:
        _rpu_execution_components = Qwen3VLAdapter._rpu_execution_components
        seen = None

        @classmethod
        def preflight(cls, _config):
            pass

        @classmethod
        def preflight_execution(cls, _config, execution):
            cls.seen = execution

        def __init__(self, model):
            self.model = model

    model = Model()
    monkeypatch.setattr(
        conditional_generation.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        conditional_generation,
        "resolve_config_architecture",
        lambda *_args, **_kwargs: "Qwen3VLForConditionalGeneration",
    )
    monkeypatch.setattr(
        conditional_generation, "get_adapter", lambda _arch: Adapter
    )
    monkeypatch.setattr(
        conditional_generation.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )

    result = conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
        "unused",
        rpu_execution={
            "vision": {"chunk_size": 144},
            "components": {
                "vision_encoder": {"vision": {"chunk_size": 256}},
            },
        },
    )

    assert result is model
    assert Adapter.seen["vision"]["chunk_size"] == 144
    assert (
        Adapter.seen["components"]["vision_encoder"]["vision"]["chunk_size"]
        == 256
    )


def test_conditional_loader_rejects_unknown_component_before_weight_load(
    monkeypatch,
):
    class Adapter:
        _rpu_execution_components = Qwen3VLAdapter._rpu_execution_components

    monkeypatch.setattr(
        conditional_generation.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        conditional_generation,
        "resolve_config_architecture",
        lambda *_args, **_kwargs: "Qwen3VLForConditionalGeneration",
    )
    monkeypatch.setattr(
        conditional_generation, "get_adapter", lambda _arch: Adapter
    )
    monkeypatch.setattr(
        conditional_generation.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("weights were loaded"),
    )

    with pytest.raises(ValueError, match="unknown component"):
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
            "unused",
            rpu_execution={"components": {"typo": {}}},
        )
