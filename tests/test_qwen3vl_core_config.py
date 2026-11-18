"""Composite VL core configuration and preclaim admission using real HF trees."""
import copy

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from rpu_backend.adapters import qwen3_vl as vl
from rpu_backend.api import _execution, conditional_generation
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.decoder import validate_qwen3_vl_text_semantics
from rpu_backend.runtime.topology import execution_core_count


_GEOMETRY = {2: (2048, 6144, 28, 16), 4: (2560, 9728, 36, 32),
             8: (4096, 12288, 36, 32)}


def _config(size=2):
    hidden, intermediate, layers, heads = _GEOMETRY[size]
    return Qwen3VLConfig(
        architectures=["Qwen3VLForConditionalGeneration"],
        tie_word_embeddings=size != 8,
        image_token_id=151655, video_token_id=151656,
        vision_start_token_id=151652, vision_end_token_id=151653,
        text_config=dict(hidden_size=hidden, intermediate_size=intermediate,
            num_hidden_layers=layers, num_attention_heads=heads,
            num_key_value_heads=8, head_dim=128, vocab_size=151936,
            hidden_act="silu", rms_norm_eps=1e-6, attention_bias=False,
            use_cache=True, tie_word_embeddings=size != 8,
            max_position_embeddings=262144,
            rope_parameters={"rope_type": "default", "rope_theta": 5_000_000.0,
                "mrope_interleaved": True, "mrope_section": [24, 20, 20]}),
        vision_config=dict(hidden_size=1152 if size == 8 else 1024,
            intermediate_size=4304 if size == 8 else 4096,
            depth=27 if size == 8 else 24, num_heads=16,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=hidden, in_channels=3, num_position_embeddings=2304,
            hidden_act="gelu_pytorch_tanh",
            deepstack_visual_indexes=[8, 16, 24] if size == 8 else [5, 11, 17]))


def _model(size=2, cores=6):
    # CPU FakeTensors preserve actual dtype/device/stride checks without
    # allocating multi-GB weights or replacing any production validator.
    with FakeTensorMode():
        model = Qwen3VLForConditionalGeneration._from_config(
            _config(size), dtype=torch.float16).eval()
    model._rpu_execution = {"model": {"num_cores": cores}}
    return model


def _weights(model):
    return tuple((name, id(value), value._version) for name, value in model.named_parameters())


@pytest.fixture(autouse=True)
def safe_process(monkeypatch):
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)


@pytest.mark.parametrize("size", (2, 4, 8))
@pytest.mark.parametrize("cores", (4, 6))
def test_exact_profiles_derive_text_budget_and_four_core_vision_without_mutation(size, cores):
    model = _model(size, cores)
    original = _weights(model)
    adapter = vl.Qwen3VLAdapter(model)
    topology = vl._qwen3_vl_core_topology(model.config, model._rpu_execution)
    assert topology.identity()[1:] == (cores, 4, cores, 4, 8)
    assert adapter._topology == topology
    assert execution_core_count(adapter._text_execution) == cores
    assert execution_core_count(adapter._vision_execution) == 4
    assert execution_core_count(model.model.language_model._rpu_execution) == cores
    assert execution_core_count(model.model.visual._rpu_execution) == 4
    assert model.model.language_model._fmb_execution_component_id == "language_model"
    assert model.model.visual._fmb_execution_component_id == "vision_encoder"
    # Exercise the actual complete text/Vision inventories after Session views
    # have been attached. Cold views must not look like converted weights.
    vl._validate_qwen3_vl_reduced_weights(model, topology)
    assert _weights(model) == original
    assert not hasattr(model, "_rpu_swizzle_started")
    assert model.config.text_config.intermediate_size == _GEOMETRY[size][1]
    assert model.model.language_model.layers[-1].mlp.up_proj.out_features == _GEOMETRY[size][1]
    with model.lm_head.weight.fake_mode:
        cache = RPUCache.from_model(model.model.language_model, max_seq_len=16, device="cpu")
    assert (cache.attn_tp, cache.physical_kv_cores) == (4, 8)
    assert cache.num_layers == _GEOMETRY[size][2]
    assert cache.nKVHeadVx * cache.nKVHeadChunk == 16
    assert not hasattr(model.model.language_model, "_rpu_decoder_topology")


@pytest.mark.parametrize("override", (
    {"model": {"num_cores": True}}, {"model": {"num_cores": 5}},
    {"model": {"num_cores": 6}, "components": {"language_model": {"model": {"num_cores": 4}}}},
))
def test_invalid_core_controls_reject_before_hf_weights(monkeypatch, override):
    config = _config()
    monkeypatch.setattr(conditional_generation.AutoConfig, "from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr(conditional_generation.AutoModelForImageTextToText, "from_pretrained",
                        lambda *a, **kw: pytest.fail("invalid core request loaded HF weights"))
    with pytest.raises(ValueError):
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
            "synthetic-vl", device="cpu", rpu_execution=override)


@pytest.mark.parametrize("fault", ("architecture", "text", "vision", "mrope", "quant", "special_tokens"))
def test_unsupported_profile_rejects_before_hf_weights(monkeypatch, fault):
    config = _config()
    if fault == "architecture":
        config.architectures = ["Qwen3VLMoeForConditionalGeneration"]
    elif fault == "text":
        config.text_config.num_hidden_layers = 16
    elif fault == "vision":
        config.vision_config.num_heads = 8
    elif fault == "mrope":
        config.text_config.rope_parameters["mrope_interleaved"] = False
    elif fault == "quant":
        config.quantization_config = {"quant_method": "awq"}
    else:
        config.image_token_id += 1
    monkeypatch.setattr(conditional_generation.AutoConfig, "from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr(conditional_generation.AutoModelForImageTextToText, "from_pretrained",
                        lambda *a, **kw: pytest.fail("unsupported profile loaded HF weights"))
    with pytest.raises((vl.UnsupportedModelError, ValueError)):
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
            "synthetic-vl", device="cpu", rpu_execution={"model": {"num_cores": 6}})


@pytest.mark.parametrize("budget_patch", ({"model": {"num_cores": 4}}, {"model": {}}))
def test_reconfigure_rejects_changed_or_removed_budget_before_any_config_publication(monkeypatch, budget_patch):
    model = _model()
    adapter = vl.Qwen3VLAdapter(model)
    adapter._rpu_is_ready = True  # Reach the real cold-topology check without native installation.
    session = model._execution_session
    configs = (model._rpu_execution, adapter._rpu_execution,
               model.model.language_model._rpu_execution, model.model.visual._rpu_execution)
    generation = session.generation
    monkeypatch.setattr(adapter, "_apply_execution_state",
                        lambda *a, **kw: pytest.fail("changed budget reached native/config mutation"))
    with pytest.raises(ValueError, match="cold-only"):
        _execution.reconfigure_rpu_execution(model, budget_patch)
    assert (model._rpu_execution, adapter._rpu_execution,
            model.model.language_model._rpu_execution, model.model.visual._rpu_execution) == configs
    assert session.generation == generation
    assert adapter._topology.num_cores == 6
    adapter._rpu_is_ready = False


class _ReachedClaim(RuntimeError):
    pass


@pytest.mark.parametrize("fault", (None, "last_text_weight", "last_vision_weight", "profile_name", "config_drift", "extra_linear", "vision_batch", "last_norm_epsilon", "last_activation"))
def test_full_to_rpu_revalidates_both_towers_before_claim(monkeypatch, fault):
    model = _model(4)
    adapter = vl.Qwen3VLAdapter(model)
    if fault == "vision_batch":
        monkeypatch.setenv("RPU_QWEN3VL_VISION_BATCH", "1")
    with model.lm_head.weight.fake_mode:
        if fault == "last_text_weight":
            leaf = model.model.language_model.layers[-1].mlp.down_proj
            leaf.weight = torch.nn.Parameter(leaf.weight[:-1])
        elif fault == "last_vision_weight":
            leaf = model.model.visual.deepstack_merger_list[-1].linear_fc2
            leaf.weight = torch.nn.Parameter(leaf.weight[:-1])
        elif fault == "profile_name":
            model._qwen3_vl_delivery_profile = "unadmitted-profile"
        elif fault == "config_drift":
            # Another valid finite profile has the same 6-core topology tuple;
            # the adapter must also keep its original geometry identity.
            model.config = copy.deepcopy(_config(2))
        elif fault == "extra_linear":
            model.model.language_model.layers[-1].extra = torch.nn.Linear(16, 16, dtype=torch.float16)
        elif fault == "last_norm_epsilon":
            model.model.language_model.layers[-1].post_attention_layernorm.variance_epsilon = 1e-5
        elif fault == "last_activation":
            model.model.language_model.layers[-1].mlp.act_fn = torch.nn.GELU()
    original = _weights(model)
    claimed = []
    def claim(value):
        claimed.append(value)
        raise _ReachedClaim("preclaim checks complete")
    monkeypatch.setattr(vl, "_claim_live_instance", claim)
    monkeypatch.setattr(vl, "convert_linear_weights_inplace",
                        lambda *a, **kw: pytest.fail("preclaim failure converted weights"))
    if fault is None:
        with pytest.raises(_ReachedClaim):
            adapter.to_rpu()
        assert claimed == [model]
    else:
        with pytest.raises((vl.UnsupportedModelError, ValueError)):
            adapter.to_rpu()
        assert claimed == []
    assert _weights(model) == original
    assert not hasattr(model, "_rpu_swizzle_started")
    assert adapter._closed is False and adapter._gc_retirement_enabled is False


def test_text_inventory_helper_checks_the_final_actual_leaf():
    model = _model(2)
    topology = vl._qwen3_vl_core_topology(model.config, model._rpu_execution)
    vl._validate_qwen3_vl_reduced_weights(model, topology)
    with model.lm_head.weight.fake_mode:
        layer = model.model.language_model.layers[-1]
        layer.self_attn.k_norm.weight = torch.nn.Parameter(layer.self_attn.k_norm.weight[:-1])
    with pytest.raises(ValueError, match="layer 27 k_norm"):
        vl._validate_qwen3_vl_reduced_weights(model, topology)


@pytest.mark.parametrize("fault", ("q_norm", "k_norm", "input_layernorm", "post_attention_layernorm", "final_norm", "activation", "rotary_type", "rotary_scale", "rotary_section", "rotary_config"))
def test_live_text_semantics_reject_late_scalar_or_module_changes(fault):
    model = _model()
    text = model.model.language_model
    validate_qwen3_vl_text_semantics(text)
    layer = text.layers[-1]
    if fault in {"q_norm", "k_norm"}:
        getattr(layer.self_attn, fault).variance_epsilon = 1e-5
    elif fault in {"input_layernorm", "post_attention_layernorm"}:
        getattr(layer, fault).variance_epsilon = 1e-5
    elif fault == "final_norm":
        text.norm.variance_epsilon = 1e-5
    elif fault == "activation":
        layer.mlp.act_fn = torch.nn.ReLU()
    elif fault == "rotary_type":
        text.rotary_emb.rope_type = "dynamic"
    elif fault == "rotary_scale":
        text.rotary_emb.attention_scaling = 2.0
    elif fault == "rotary_section":
        text.rotary_emb.mrope_section = [32, 16, 16]
    else:
        text.rotary_emb.config = copy.deepcopy(text.config)
        text.rotary_emb.config.rope_parameters["rope_theta"] = 10000.0
    with pytest.raises(ValueError):
        validate_qwen3_vl_text_semantics(text)


def test_preinstall_vl_cache_rejects_changed_mrope_before_allocation(monkeypatch):
    model = _model()
    vl.Qwen3VLAdapter(model)
    text = model.model.language_model
    text.config.rope_parameters["mrope_section"] = [32, 16, 16]
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: pytest.fail("invalid VL profile allocated cache"))
    with pytest.raises(ValueError, match="exact dense Qwen3-VL"):
        RPUCache.from_model(text, max_seq_len=16, device="cpu")


def _w8_config():
    from rpu_backend.quant.load import _QWEN3_VL_2B_W8A16_QUANT_CONFIG
    config = _config()
    config.quant_config = copy.deepcopy(_QWEN3_VL_2B_W8A16_QUANT_CONFIG)
    return config


def _w8_model(cores):
    model = _model(2, cores)
    model.config.quant_config = _w8_config().quant_config
    with model.lm_head.weight.fake_mode:
        for layer in model.model.language_model.layers:
            for role in vl._QWEN3_VL_TEXT_PROJECTIONS:
                linear = layer.get_submodule(role)
                linear.weight = torch.nn.Parameter(linear.weight.to(torch.int8), requires_grad=False)
                linear.register_buffer("weight_scale", torch.ones(linear.out_features, dtype=torch.float16))
    return model


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_w8_text_profile_preserves_fp16_vision_tied_head_and_core_topology(cores):
    from rpu_backend.runtime.decoder import _validate_mrope_decoder_topology
    model = _w8_model(cores)
    original = _weights(model)
    adapter = vl.Qwen3VLAdapter(model)
    assert adapter._is_text_w8a16_2b and not adapter._is_graph_blocked_32b_w8a16
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert model.lm_head.weight.dtype == torch.float16
    assert all(p.dtype == torch.float16 for p in model.model.visual.parameters())
    topology = adapter._topology
    if cores != 8:
        assert topology.identity() == (2, cores, 4, cores, 4, 8)
        text = model.model.language_model
        _validate_mrope_decoder_topology(text, config=text.config,
            arch="qwen3_vl_text", execution_config=adapter._text_execution,
            topology=topology, mrope_section=(24, 20, 20),
            deepstack_lang_layers=(0, 1, 2), scale_lists=vl._qwen3_vl_text_scale_lists(text))
    assert _weights(model) == original
    assert not hasattr(model, "_rpu_swizzle_started")


@pytest.mark.parametrize("fault", ("last_scale", "last_weight", "vision", "head", "untie", "config", "norm", "extra_linear"))
def test_w8_text_inventory_is_rechecked_before_claim(monkeypatch, fault):
    model = _w8_model(6)
    adapter = vl.Qwen3VLAdapter(model)
    with model.lm_head.weight.fake_mode:
        last = model.model.language_model.layers[-1]
        if fault == "last_scale":
            last.mlp.down_proj.weight_scale = last.mlp.down_proj.weight_scale[:-1]
        elif fault == "last_weight":
            last.self_attn.q_proj.weight = torch.nn.Parameter(last.self_attn.q_proj.weight.half())
        elif fault == "vision":
            model.model.visual.deepstack_merger_list[-1].linear_fc2.weight = torch.nn.Parameter(
                model.model.visual.deepstack_merger_list[-1].linear_fc2.weight.to(torch.int8), requires_grad=False)
        elif fault == "head":
            model.lm_head.register_buffer("weight_scale", torch.ones(151936, dtype=torch.float16))
        elif fault == "untie":
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
        elif fault == "config":
            model.config.quant_config["qaxis"] = 1
        elif fault == "norm":
            last.self_attn.q_norm.weight = torch.nn.Parameter(last.self_attn.q_norm.weight[:-1])
        else:
            last.extra = torch.nn.Linear(16, 16, dtype=torch.float16)
    monkeypatch.setattr(vl, "_claim_live_instance", lambda *_: pytest.fail("bad W8 tree claimed RPU"))
    with pytest.raises((ValueError, vl.RPUBackendError)):
        adapter.to_rpu()
    assert not hasattr(model, "_rpu_swizzle_started")


def test_w8_native_install_boundary_rejects_a_foreign_scale_tensor():
    from rpu_backend.runtime.decoder import _validate_mrope_decoder_topology
    model = _w8_model(6)
    adapter = vl.Qwen3VLAdapter(model)
    text = model.model.language_model
    scales = vl._qwen3_vl_text_scale_lists(text)
    with model.lm_head.weight.fake_mode:
        scales[-1][-1] = scales[-1][-1].clone()
    with pytest.raises(ValueError, match="actual INT8 weight / FP16 per-channel scale"):
        _validate_mrope_decoder_topology(text, config=text.config,
            arch="qwen3_vl_text", execution_config=adapter._text_execution,
            topology=adapter._topology, mrope_section=(24, 20, 20),
            deepstack_lang_layers=(0, 1, 2), scale_lists=scales)


def test_w8_loader_restores_omitted_tied_head_without_casting_int8(monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from rpu_backend.quant import load as loader
    from rpu_backend.tests.library.test_w8a16_loader_meta import (
        _TinyImageText, _tiny_imagetext_checkpoint,
    )
    def skeleton(_config):
        model = _TinyImageText()
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
        return model
    monkeypatch.setattr(loader.AutoModelForImageTextToText, "from_config", skeleton)
    tensors = _tiny_imagetext_checkpoint()
    del tensors["lm_head.weight"]
    save_file(tensors, str(tmp_path / "model.safetensors"))
    model = loader.load_w8a16_imagetext(_w8_config(), str(tmp_path))
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert model.lm_head.weight.dtype == torch.float16
    q = model.model.language_model.layers[0].self_attn.q_proj
    assert q.weight.dtype == torch.int8 and q.weight_scale.dtype == torch.float16
    assert all(t.device.type == "cpu" for t in (*model.parameters(), *model.buffers()))
    with pytest.raises(ValueError, match="metadata-only staging requires exact Qwen3-VL-4B/32B"):
        loader.stage_w8a16_imagetext_for_rpu(_w8_config(), str(tmp_path))


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_public_w8_loader_uses_cpu_inventory_before_any_device_claim(monkeypatch, cores):
    from rpu_backend.quant import load as loader
    model = _w8_model(cores)
    calls = []
    monkeypatch.setattr(conditional_generation.AutoConfig, "from_pretrained", lambda *a, **kw: model.config)
    def load(config, path, **kwargs):
        calls.append((config, path, kwargs))
        return model
    monkeypatch.setattr(loader, "load_w8a16_imagetext", load)
    monkeypatch.setattr(loader, "stage_w8a16_imagetext_for_rpu", lambda *a, **kw: pytest.fail("2B used 32B staging"))
    monkeypatch.setattr(conditional_generation.AutoModelForImageTextToText, "from_pretrained",
                        lambda *a, **kw: pytest.fail("INT8 checkpoint reached casting HF loader"))
    result = conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
        "synthetic-w8", device="cpu", rpu_execution={"model": {"num_cores": cores}})
    assert result is model and len(calls) == 1
    assert not hasattr(model, "_rpu_swizzle_started")
