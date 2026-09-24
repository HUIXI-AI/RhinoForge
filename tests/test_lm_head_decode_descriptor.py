"""Actual composite install/planner/runner with a mutable native topology double."""
import pytest
import torch

from rpu_backend.adapters import qwen3_vl as qvl
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime import decoder
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import (
    _Model, _decode_descriptor, install_runtime,  # noqa: F401
)


def _model(size):
    model = _Model()
    cfg = model.config
    text, vision = cfg.text_config, cfg.vision_config
    if size in (4, 8):
        text.hidden_size, text.intermediate_size = (2560, 9728) if size == 4 else (4096, 12288)
        text.num_hidden_layers = 36
        text.num_attention_heads = 32
        vision.out_hidden_size = text.hidden_size
    if size == 8:
        cfg.tie_word_embeddings = False
        text.model_type = "qwen3_vl_text"
        text.hidden_act = "silu"
        text.rms_norm_eps = 1e-6
        text.vocab_size = 151936
        text.attention_bias = False
        text.use_cache = True
        text.rope_parameters = dict(rope_type="default", rope_theta=5_000_000.0,
                                   mrope_interleaved=True, mrope_section=[24, 20, 20])
        vision.hidden_size, vision.intermediate_size, vision.depth = 1152, 4304, 27
        vision.deepstack_visual_indexes = [8, 16, 24]
        vision.model_type, vision.in_channels, vision.num_position_embeddings = "qwen3_vl", 3, 2304
        model.model.language_model.layers = []  # Exercise staging admission without any model-sized allocation.
    return model


@pytest.mark.parametrize("size", [2, 4, 8])
def test_vl_final_lm_head_topology_is_planned_before_first_decode(install_runtime, monkeypatch, size):
    model = _model(size)
    adapter = qvl.Qwen3VLAdapter(model)
    assert adapter.to_rpu() is model
    text = model.model.language_model
    assert install_runtime["decode_plans"] == [False, True]
    assert text._rpu_decode_stage_descriptor == _decode_descriptor(True)
    assert text._decoder_decode_plan[1].selected.stage_tuple.physical_descriptor == _decode_descriptor(True)
    assert adapter.to_rpu() is model
    assert install_runtime["decode_plans"] == [False, True]

    dispatched = []

    def forward(handle, hidden, keys, values, mask, position, causal, positions, visual, *rest):
        # This models the same native family-9/site check that rejected the
        # real board's first decode. No fallback permits the old topology.
        descriptor = tuple(rest[-1])
        assert descriptor == _decode_descriptor(install_runtime["lm_head_set"])
        assert position == 314
        dispatched.append(descriptor)
        return hidden.clone()

    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_get_resolved_chunk_size", lambda handle: 16, raising=False)
    cache = RPUCache(1, 1, 384, 1, 16, device="cpu")
    cache.reset_to_position(314)
    decoder._run_causal_decoder_forward(
        text, text._rpu_decoder_handle, inputs_embeds=torch.ones(1, 1, 2).half(),
        past_key_values=cache, use_cache=True, return_dict=True,
    )
    assert dispatched == [_decode_descriptor(True)] and cache.position == 315
    adapter.close()


@pytest.mark.parametrize("size", [2, 4, 8])
def test_failed_final_decode_plan_retires_both_children_before_ready(install_runtime, size):
    model = _model(size)
    adapter = qvl.Qwen3VLAdapter(model)
    install_runtime["plan_error"] = error = RuntimeError("final topology planner rejected")
    with pytest.raises(RuntimeError) as caught:
        adapter.to_rpu()
    assert caught.value is error
    assert install_runtime["decode_plans"] == [False, True]
    assert not adapter._rpu_is_ready and not model._rpu_swizzled
    assert not hasattr(model.model.language_model, "_rpu_decoder_handle")
    assert not hasattr(model.model.visual, "_rpu_vision_handle")
    assert "forward" not in vars(model)
    assert install_runtime["events"][-2:] == [("destroy", "text"), ("destroy", "vision")]
    with pytest.raises(qvl.RPUBackendError, match="prior swizzle attempt"):
        adapter.to_rpu()
    assert install_runtime["decode_plans"] == [False, True]
