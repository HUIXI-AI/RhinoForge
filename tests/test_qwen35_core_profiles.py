"""Cold admission and cache boundaries for actual dense Qwen3.5 profiles."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3_5Config

from rpu_backend.adapters.qwen3_5.cores import (
    core_topology, topology_tuple, validate_core_profile,
)
from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache


@pytest.mark.parametrize("size", ("0_8", "2", "4", "9"))
@pytest.mark.parametrize("cores", (4, 6))
def test_dense_text_example_core_profile_is_exact(size, cores, tmp_path):
    from pathlib import Path
    from examples.common.config import ConfigError, load_config, planner_profiles

    path = Path(__file__).resolve().parents[1] / f"examples/configs/qwen3_5/text/{size}b/fp16_tp{cores}.toml"
    config = load_config(path)
    assert config["rpu_execution"]["model"]["num_cores"] == cores
    profile = planner_profiles()[config["example"]["profile_id"]]
    assert profile["support"]["status"] == "IMPLEMENTED_UNVERIFIED_HW"
    for other in (10 - cores, 8):
        bad = tmp_path / f"bad{other}.toml"
        bad.write_text(path.read_text().replace(f"num_cores = {cores}", f"num_cores = {other}"))
        with pytest.raises(ConfigError, match="exact fixed profile"):
            load_config(bad)


def _config(size=2):
    h, inter, layers, q, kv, value_heads = {
        0.8: (1024, 3584, 24, 8, 2, 16),
        2: (2048, 6144, 24, 8, 2, 16),
        4: (2560, 9216, 32, 16, 4, 32),
        9: (4096, 12288, 32, 16, 4, 32),
    }[size]
    return Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        image_token_id=248056, tie_word_embeddings=size != 9,
        text_config=dict(hidden_size=h, intermediate_size=inter, num_hidden_layers=layers,
            num_attention_heads=q, num_key_value_heads=kv, head_dim=256, vocab_size=248320,
            layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(layers)],
            attention_bias=False, hidden_act="silu", attn_output_gate=True, use_cache=True,
            full_attention_interval=4, linear_conv_kernel_dim=4, linear_num_key_heads=16,
            linear_num_value_heads=value_heads, linear_key_head_dim=128, linear_value_head_dim=128,
            rms_norm_eps=1e-6, rope_parameters=dict(rope_type="default", rope_theta=10_000_000,
                partial_rotary_factor=0.25, mrope_interleaved=True, mrope_section=[11, 11, 10])),
        vision_config=dict(depth=12 if size == 0.8 else 27 if size == 9 else 24,
            hidden_size=768 if size == 0.8 else 1152 if size == 9 else 1024,
            intermediate_size=3072 if size == 0.8 else 4304 if size == 9 else 4096,
            num_heads=12 if size == 0.8 else 16, out_hidden_size=h,
            in_channels=3, patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
            num_position_embeddings=2304, hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[]))


@pytest.mark.parametrize("size", [0.8, 2, 4, 9])
@pytest.mark.parametrize("cores, expected", [(4, (4, 4, 4, 4, 8)),
    (6, (6, 4, 6, 4, 8)), (8, (8, 8, 8, 8, 8))])
def test_official_shapes_resolve_one_cold_role_template(size, cores, expected):
    config = _config(size)
    # The real checkpoints carry BF16 metadata even when loaded as FP16.
    config.text_config.dtype = torch.bfloat16
    result = validate_core_profile(config, {"model": {"num_cores": cores}})
    assert topology_tuple(result) == expected
    from rpu_backend.runtime.topology import decoder_mlp_intermediate_size
    physical = decoder_mlp_intermediate_size(config.text_config.intermediate_size, result.mlp_tp)
    assert physical % (result.mlp_tp * 32) == 0
    assert physical == (3648 if size == 0.8 and cores == 6 else config.text_config.intermediate_size)


@pytest.mark.parametrize("where, name, value", [
    ("text", "linear_num_key_heads", 8), ("text", "linear_num_value_heads", 24),
    ("text", "head_dim", 128), ("text", "num_attention_heads", True),
    ("text", "attention_bias", True), ("text", "attn_output_gate", False),
    ("text", "num_experts", 4), ("text", "quantization_config", {"bits": 8}),
    ("text", "adaptive_mode", "adaLN"), ("text", "vocab_size", 151936),
    ("vision", "out_hidden_size", 2560), ("vision", "temporal_patch_size", 1),
    ("vision", "deepstack_visual_indexes", [5]), ("vision", "hidden_act", "gelu"),
    ("root", "tie_word_embeddings", False), ("root", "image_token_id", 248057),
])
def test_reduced_profile_rejects_semantic_near_neighbours(where, name, value):
    config = _config()
    owner = {"text": config.text_config, "vision": config.vision_config, "root": config}[where]
    setattr(owner, name, value)
    with pytest.raises(ValueError):
        validate_core_profile(config, {"model": {"num_cores": 6}})


@pytest.mark.parametrize("mutation", ["layer_order", "rope_section", "partial", "rope_theta"])
def test_reduced_profile_does_not_admit_a_shape_only_match(mutation):
    config = _config()
    tc = config.text_config
    if mutation == "layer_order":
        tc.layer_types[0], tc.layer_types[3] = tc.layer_types[3], tc.layer_types[0]
    else:
        key, value = {"rope_section": ("mrope_section", [10, 11, 11]),
            "partial": ("partial_rotary_factor", 0.5), "rope_theta": ("rope_theta", 1_000_000)}[mutation]
        tc.rope_parameters[key] = value
    with pytest.raises(ValueError):
        validate_core_profile(config, {"model": {"num_cores": 4}})


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("size", [0.8, 2])
@pytest.mark.parametrize("entry", ["conditional", "causal", "text"])
def test_cache_factory_uses_compute_heads_but_eight_physical_controllers(cores, size, entry):
    config = _config(size)
    execution = {"model": {"num_cores": cores}}
    topology = core_topology(execution)
    # The installed text child owns state.execution_config, while only the
    # public wrapper publishes _rpu_execution.
    inner = SimpleNamespace(config=config.text_config, _rpu_qwen3_5=SimpleNamespace(
        execution_topology=topology, execution_config=execution))
    model = {
        "conditional": SimpleNamespace(config=config, _rpu_execution=execution,
            model=SimpleNamespace(language_model=inner)),
        "causal": SimpleNamespace(config=config.text_config, _rpu_execution=execution, model=inner),
        "text": inner,
    }[entry]
    cache = Qwen3_5Cache.from_model(model, 65, device="cpu")
    assert cache.execution_topology == topology_tuple(topology)
    assert cache.num_kv_heads == cache.attn_tp == topology.attn_tp
    assert cache.physical_kv_cores == cache.nKVHeadChunk == 8
    assert cache.allocated_max_seq_len == 192
    assert cache.conv_states[0].shape == (topology.attn_tp, 4, 6144 // topology.attn_tp)
    assert cache.recurrent_states[0].shape == (16, 128, 128)
    assert cache.k_caches[0].numel() == 0 and cache.k_caches[3].numel() > 0
    pointers = [x.data_ptr() for x in (cache.conv_states[0], cache.recurrent_states[0], cache.k_caches[3])]
    cache.conv_states[0].fill_(1)
    cache.recurrent_states[0].fill_(1)
    cache.reset()
    assert pointers == [x.data_ptr() for x in (cache.conv_states[0], cache.recurrent_states[0], cache.k_caches[3])]
    assert not cache.conv_states[0].count_nonzero() and not cache.recurrent_states[0].count_nonzero()


def test_cache_factory_rejects_changed_or_uninstalled_core_geometry():
    config = _config()
    state = SimpleNamespace(execution_topology=core_topology({"model": {"num_cores": 4}}))
    model = SimpleNamespace(config=config, _rpu_execution={"model": {"num_cores": 6}},
        model=SimpleNamespace(language_model=SimpleNamespace(_rpu_qwen3_5=state)))
    with pytest.raises(ValueError, match="no longer matches"):
        Qwen3_5Cache.from_model(model, 64, device="cpu")
    del state.execution_topology
    with pytest.raises(ValueError, match="completely installed"):
        Qwen3_5Cache.from_model(model, 64, device="cpu")


def test_child_cache_rejects_changed_installed_execution_before_allocation(monkeypatch):
    state = SimpleNamespace(execution_topology=core_topology({"model": {"num_cores": 4}}),
        execution_config={"model": {"num_cores": 6}})
    inner = SimpleNamespace(config=_config().text_config, _rpu_qwen3_5=state)
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: pytest.fail("stale topology allocated cache"))
    with pytest.raises(ValueError, match="no longer matches"):
        Qwen3_5Cache.from_model(inner, 64, device="cpu")


def test_core_change_rejection_precedes_native_transaction(monkeypatch):
    import rpu_backend.adapters.qwen3_5 as adapter

    config = _config()
    state = SimpleNamespace(execution_topology=core_topology({"model": {"num_cores": 4}}))
    model = SimpleNamespace(config=config)
    owner = SimpleNamespace(model=model, preflight_execution=adapter.Qwen3_5Adapter.preflight_execution)
    monkeypatch.setattr(adapter, "_qwen3_5_text_runtime_state", lambda _: state)
    monkeypatch.setattr(adapter, "torch", SimpleNamespace())
    with pytest.raises(ValueError, match="immutable"):
        adapter._validate_qwen3_5_execution_reconfigure(owner, {"model": {"num_cores": 6}})


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("size", [0.8, 4])
def test_public_loader_admits_exact_profile_before_loading_weights(monkeypatch, invalid, size):
    import rpu_backend.adapters.qwen3_5 as adapter
    from rpu_backend.api import conditional_generation

    config = _config(size)
    if invalid:
        config.text_config.linear_num_value_heads = 24
    monkeypatch.setattr(conditional_generation.AutoConfig, "from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr(conditional_generation, "get_adapter", lambda _: adapter.Qwen3_5Adapter)
    loaded = []

    class ReachedWeights(Exception):
        pass

    def load(*args, **kwargs):
        loaded.append(kwargs["config"])
        raise ReachedWeights

    monkeypatch.setattr(adapter.Qwen3_5Adapter, "load_pretrained", load)
    with pytest.raises(ValueError if invalid else ReachedWeights):
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
            "/never-read-weights", rpu_execution={"model": {"num_cores": 6}})
    assert loaded == ([] if invalid else [config])


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("linear_acc32", [False, True])
def test_public_forward_lm_head_dispatch_uses_installed_accumulation(monkeypatch, cores, linear_acc32):
    import rpu_backend.adapters.qwen3_5 as adapter
    from rpu_backend.runtime import hw_attrs

    topology = core_topology({"model": {"num_cores": cores}})
    inner = SimpleNamespace(_rpu_qwen3_5=SimpleNamespace(
        execution_topology=topology, linear_acc32=linear_acc32))
    model = SimpleNamespace(model=SimpleNamespace(language_model=inner),
                            _lm_head_w_rpu_keepalive=torch.arange(32).reshape(8, 4).half())
    hidden = torch.arange(8).reshape(1, 2, 4).half()
    cache_type = type("Cache", (), {"position": 1})
    monkeypatch.setattr(adapter, "Qwen3_5Cache", cache_type)
    monkeypatch.setattr(adapter, "_preflight_qwen3_5_text", lambda *a, **kw: None)
    monkeypatch.setattr(adapter, "run_qwen3_5_text", lambda *a, **kw: hidden)
    monkeypatch.setattr(hw_attrs, "mark_first_forward_done", lambda _: None)
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    seen = []

    # A conflicting legacy global must not alter this installed model policy.
    monkeypatch.setenv("RPU_LINEAR_ACC32", "0" if linear_acc32 else "1")

    def linear(x, weight, bias, partition, num_cores, selected_acc32):
        seen.append((tuple(x.shape), partition, num_cores, selected_acc32,
                     weight is model._lm_head_w_rpu_keepalive))
        return torch.nn.functional.linear(x, weight, bias)

    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", linear, raising=False)
    output = adapter._rpu_qwen3_5_forward.__wrapped__(model, inputs_embeds=hidden,
        past_key_values=cache_type(), logits_to_keep=1)
    assert seen == [((1, 4), 1, topology.lm_head_tp, linear_acc32, True)]
    expected = torch.nn.functional.linear(hidden[:, -1:], model._lm_head_w_rpu_keepalive).float()
    assert torch.equal(output.logits, expected)


def test_reduced_batch_rejection_precedes_embedding_or_planning(monkeypatch):
    import rpu_backend.adapters.qwen3_5 as adapter
    from rpu_backend.runtime import hw_attrs

    topology = core_topology({"model": {"num_cores": 6}})
    inner = SimpleNamespace(_rpu_qwen3_5=SimpleNamespace(execution_topology=topology))
    model = SimpleNamespace(model=SimpleNamespace(language_model=inner))
    cache_type = type("Cache", (), {"position": 0})
    monkeypatch.setattr(adapter, "Qwen3_5Cache", cache_type)
    monkeypatch.setattr(hw_attrs, "mark_first_forward_done", lambda _: None)
    with pytest.raises(NotImplementedError, match="batch size 1"):
        adapter._rpu_qwen3_5_forward.__wrapped__(model, input_ids=torch.ones(2, 4, dtype=torch.long),
                                              past_key_values=cache_type())


@pytest.fixture
def graph_factories(monkeypatch):
    """Observe real CPU Graph construction at the adapter's resource boundary."""
    import rpu_backend

    calls = []
    for name in ("GraphCache", "Graph"):
        factory = getattr(rpu_backend.graph, name)

        def observe(*args, _factory=factory, _name=name, **kwargs):
            result = _factory(*args, **kwargs)
            calls.append((_name, kwargs, result))
            return result

        monkeypatch.setattr(rpu_backend.graph, name, observe)
    return calls


@pytest.fixture
def text_install_boundary(monkeypatch):
    from rpu_backend.adapters.qwen3_5 import cores, text

    # Profile validation remains real; stop before any checkpoint weight access.
    monkeypatch.setattr(cores, "validate_text_weight_geometry", lambda *args: None)
    for name in ("qwen3_5_set_execution_cores", "qwen3_5_get_execution_topology",
                 "qwen3_5_set_linear_acc32"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *args: None, raising=False)

    def stop_at_weights(model, config, **kwargs):
        state = kwargs["pending_state"]
        state.execution_topology = validate_core_profile(config, kwargs["execution_config"])
        model._rpu_qwen3_5 = state
        return 7

    monkeypatch.setattr(text, "_install_qwen3_5_text_for_rpu_impl", stop_at_weights)
    return text


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("explicit", [False, True])
def test_text_prefill_decode_graphs_keep_the_owner_policy(
    monkeypatch, graph_factories, text_install_boundary, cores, explicit,
):
    from rpu_backend.graph import GraphRuntimePolicy

    text = text_install_boundary
    monkeypatch.setenv("RPU_FASTREPLAY_SKIP_SYNC", "0")
    policy = GraphRuntimePolicy(execution_core_count=cores) if explicit else None
    model = SimpleNamespace()
    assert text.install_qwen3_5_text_for_rpu(model, _config(),
        execution_config={"model": {"num_cores": cores}},
        _graph_runtime_policy=policy) == 7
    state = model._rpu_qwen3_5
    assert state.graph_cache.runtime_policy.execution_core_count == cores
    assert state.prefill_graph.runtime_policy.execution_core_count == cores
    installed_skip_sync = state.graph_cache.runtime_policy.fast_replay_skip_sync
    if not explicit:
        assert installed_skip_sync is False
    if cores != 8 or explicit:
        assert state.graph_cache.runtime_policy is state.prefill_graph.runtime_policy
        assert graph_factories[0][1].get("max_entries") == (1 if explicit else None)
        if explicit:
            assert state.graph_cache.runtime_policy is policy
    else:
        assert all(options == {} for _, options, _ in graph_factories[:2])
    assert state.public_retained_prefill is True
    assert state.prefill_graph_cache.runtime_policy is state.graph_cache.runtime_policy
    assert graph_factories[-1][1]["max_entries"] == 1

    # Graphs created for logical Tensor inspection retain the installed policy.
    monkeypatch.setenv("RPU_FASTREPLAY_SKIP_SYNC", "1")
    debug = text._qwen35_debug_prefill_cache(state)
    assert debug.runtime_policy.execution_core_count == cores
    assert debug.runtime_policy is state.graph_cache.runtime_policy
    assert debug.runtime_policy.fast_replay_skip_sync is installed_skip_sync
    assert text._qwen35_debug_prefill_cache(state) is debug
    assert len(graph_factories) == 4


@pytest.mark.parametrize("cores, policy_cores", [(4, 8), (6, 4), (8, 6)])
def test_text_policy_conflict_rejects_before_graphs_or_weights(
    graph_factories, text_install_boundary, cores, policy_cores,
):
    from rpu_backend.graph import GraphRuntimePolicy

    model = SimpleNamespace()
    with pytest.raises(ValueError, match="graph policy conflicts"):
        text_install_boundary.install_qwen3_5_text_for_rpu(model, _config(),
            execution_config={"model": {"num_cores": cores}},
            _graph_runtime_policy=GraphRuntimePolicy(execution_core_count=policy_cores))
    assert not graph_factories and not vars(model)


@pytest.fixture
def vision_install_boundary(monkeypatch):
    from rpu_backend.adapters.qwen3_5 import vision
    from rpu_backend.api import causal_lm

    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    monkeypatch.setenv("RPU_FASTREPLAY_SKIP_SYNC", "0")
    monkeypatch.setenv("RPU_VISION_STEP0", "0")
    monkeypatch.setenv("RPU_VISION_MERGER", "0")
    events = []
    for name in ("create", "destroy", "set_execution_cores", "set_chunk_size_cap",
                 "set_cache_capacity", "set_prefill_chunk_size", "set_linear_acc32",
                 "resolve_prefill_domain", "set_weights", "set_rope",
                 "position_idx_keepalive"):
        def native(*args, _name=name):
            events.append(_name)
            return 23
        monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_" + name, native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_get_execution_topology",
                        lambda _: [4, 4, 4, 8], raising=False)
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    tensor = torch.ones(1, dtype=torch.float16)

    def convert(*args, **kwargs):
        events.append("weight-transform")
        # Policy must already be frozen before the irreversible weight boundary.
        monkeypatch.setenv("RPU_FASTREPLAY_SKIP_SYNC", "1")

    monkeypatch.setattr(vision, "_convert_vision_block_weights_for_rpu", convert)
    monkeypatch.setattr(vision, "build_vision_rope_tables", lambda *a, **kw: (tensor, tensor))
    monkeypatch.setattr(vision, "_make_dummy_vision_kv_caches", lambda *a, **kw: [])
    block = SimpleNamespace(
        attn=SimpleNamespace(proj=SimpleNamespace(weight=tensor)),
        mlp=SimpleNamespace(linear_fc1=SimpleNamespace(weight=tensor),
                            linear_fc2=SimpleNamespace(weight=tensor)),
        norm1=SimpleNamespace(weight=tensor, bias=tensor),
        norm2=SimpleNamespace(weight=tensor, bias=tensor))
    for name in ("q_w", "k_w", "v_w", "q_b", "k_b", "v_b", "o_b", "fc1_b", "fc2_b"):
        setattr(block, "_rpu_" + name, tensor)
    model = SimpleNamespace(config=_config().vision_config,
        blocks=[block] * 24, merger=torch.nn.Identity(), pos_embed=torch.nn.Identity(),
        rotary_pos_emb=SimpleNamespace(inv_freq=tensor),
        patch_embed=SimpleNamespace(proj=SimpleNamespace(
            weight=tensor.reshape(1, 1, 1, 1, 1), bias=tensor)))
    yield vision, model, events
    if hasattr(model, "_rpu_vision_handle"):
        vision._rollback_qwen3_5_vision_install(model)


@pytest.mark.parametrize("root_cores, vision_cores", [(4, 4), (6, 4), (8, 8)])
def test_vision_cached_and_debug_graphs_use_the_resolved_owner_policy(
    graph_factories, vision_install_boundary, root_cores, vision_cores,
):
    vision, model, events = vision_install_boundary
    assert vision.install_qwen3_5_vision_for_rpu(model,
        execution_config={"model": {"num_cores": root_cores}}) == 23
    cache = model._rpu_vision_graph_cache
    debug = model._rpu_vision_debug_graph
    assert cache.runtime_policy.execution_core_count == vision_cores
    assert debug.runtime_policy.execution_core_count == vision_cores
    if vision_cores != 8:
        assert cache.runtime_policy is debug.runtime_policy
        assert not cache.runtime_policy.fast_replay_skip_sync
    else:
        assert [options for _, options, _ in graph_factories] == [{"max_entries": 2}, {}]
    assert "weight-transform" in events and "set_weights" in events


def test_vision_invalid_cold_policy_rejects_before_handle_or_weights(
    monkeypatch, graph_factories, vision_install_boundary,
):
    vision, model, events = vision_install_boundary
    original_fields = dict(vars(model))
    monkeypatch.setenv("RPU_FASTREPLAY_SKIP_SYNC", "invalid")
    with pytest.raises(ValueError, match="RPU_FASTREPLAY_SKIP_SYNC"):
        vision.install_qwen3_5_vision_for_rpu(model,
            execution_config={"model": {"num_cores": 6}})
    assert not events and not graph_factories
    assert vars(model) == original_fields


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("mutation", [None, "rope", "vision", "quant"])
@pytest.mark.parametrize("vision_opt_in", [False, True])
def test_9b_cold_opt_in_uses_exact_profile_even_at_eight_cores(monkeypatch, cores, mutation, vision_opt_in):
    from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter

    config = _config(9)
    if vision_opt_in:
        monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    else:
        monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    if mutation == "rope":
        config.text_config.rope_parameters["partial_rotary_factor"] = 0.5
    elif mutation == "vision":
        config.vision_config.temporal_patch_size = 1
    elif mutation == "quant":
        config.text_config.quantization_config = {"bits": 8}
    if mutation:
        with pytest.raises(ValueError):
            Qwen3_5Adapter.preflight_execution(config, {"model": {"num_cores": cores}})
    else:
        Qwen3_5Adapter.preflight_execution(config, {"model": {"num_cores": cores}})


@pytest.fixture(scope="module")
def raw_9b_geometry():
    from transformers import Qwen3_5ForConditionalGeneration

    # Real HF modules and shapes, with no checkpoint allocation or device access.
    with torch.device("meta"):
        return Qwen3_5ForConditionalGeneration(_config(9))


def test_9b_raw_geometry_matches_real_hf_text_vision_and_untied_head(raw_9b_geometry, monkeypatch):
    from rpu_backend.adapters.qwen3_5 import cores

    checked = []
    monkeypatch.setattr(cores, "validate_cold_weight_dtype", lambda owner: checked.append(owner))
    cores.validate_dense_9b_vl_weights(raw_9b_geometry)
    assert checked == [raw_9b_geometry]
    layer = raw_9b_geometry.model.visual.blocks[0].mlp.linear_fc1
    monkeypatch.setattr(layer, "weight", torch.nn.Parameter(torch.empty(4303, 1152, device="meta")))
    with pytest.raises(ValueError, match="requires shape"):
        cores.validate_dense_9b_vl_weights(raw_9b_geometry)


@pytest.mark.parametrize("device,dtype,fp16,accepted", [
    ("cpu", torch.float16, True, True), ("cpu", torch.bfloat16, False, True),
    ("cpu", torch.float32, False, True), ("cpu", torch.float32, True, False),
    ("cpu", torch.int8, False, False), ("meta", torch.float16, False, False),
])
def test_9b_actual_weight_dtype_and_device_boundary(device, dtype, fp16, accepted):
    from rpu_backend.adapters.qwen3_5.cores import validate_cold_weight_dtype

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(
        torch.empty(1, device=device, dtype=dtype), requires_grad=dtype.is_floating_point))
    if accepted:
        validate_cold_weight_dtype(model, fp16=fp16)
    else:
        with pytest.raises(ValueError, match="cold weight"):
            validate_cold_weight_dtype(model, fp16=fp16)


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("fault", ["geometry", "late_pool", "no_device", None])
@pytest.mark.parametrize("vision_opt_in", [False, True])
def test_9b_cold_arena_preparation_precedes_any_weight_mutation(monkeypatch, cores, fault, vision_opt_in):
    from rpu_backend.adapters import qwen3_5 as adapter
    from rpu_backend.adapters.qwen3_5 import cores as profiles
    from rpu_backend.api._execution import bind_execution_session
    from rpu_backend.graph import GraphRuntimePolicy

    events = []
    config = _config(9)

    class Model:
        def __init__(self):
            self.config = config
            self.model = SimpleNamespace(language_model=SimpleNamespace(config=config.text_config))
            self.forward = object()

        def get_output_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(shape=(248320, 4096)))

        def half(self):
            events.append("weight-mutation")
            raise RuntimeError("stop at mutation boundary")

    model = Model()
    owner = adapter.Qwen3_5Adapter.__new__(adapter.Qwen3_5Adapter)
    owner.model, owner._rpu_is_ready = model, False
    execution = {"model": {"num_cores": cores}}
    owner._execution_session = bind_execution_session(
        model, execution, entry_point="9B cold install test",
        graph_mode="RETAINED_CACHE" if cores == 8 and not vision_opt_in else "BOUNDED_ONESHOT",
        supported=adapter.Qwen3_5Adapter._EXECUTION_SUPPORTED)
    owner._execution_session._bind_planner_owner(model.model.language_model)
    model._rpu_execution = owner._execution_session.config
    if vision_opt_in:
        monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    else:
        monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    monkeypatch.setattr(adapter, "_claim_live_instance", lambda _: events.append("claim"))
    monkeypatch.setattr(profiles, "validate_text_weight_geometry", lambda *a: None)
    for name in ("qwen3_5_set_execution_cores", "qwen3_5_get_execution_topology",
                 "qwen3_5_vision_set_execution_cores", "qwen3_5_vision_get_execution_topology",
                 "linear_with_accumulation"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *a: None, raising=False)

    def geometry(value):
        assert value is model
        events.append("geometry")
        if fault == "geometry":
            raise ValueError("invalid actual geometry")

    def prepare(policy):
        assert policy.execution_core_count == cores and policy.graph_arena_count == 4
        assert not policy.qwen35_legacy_27b_sdk_budget
        events.append("prepare-pool")
        if fault == "late_pool":
            raise RuntimeError("SDK pool frozen")
        return fault != "no_device"

    monkeypatch.setattr(profiles, "validate_dense_9b_vl_weights", geometry)
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", prepare)
    with pytest.raises((ValueError, RuntimeError, adapter.RPUBackendError)):
        owner.to_rpu(max_seq_len=128)
    expected = ["geometry"] if fault == "geometry" else ["geometry", "claim", "prepare-pool"]
    if fault is None:
        expected.append("weight-mutation")
    assert events == expected
    assert getattr(model, "_rpu_swizzle_started", False) is (fault is None)


@pytest.mark.parametrize("cores,vision_cores", [(4, 4), (6, 4), (8, 8)])
def test_9b_text_and_lazy_vision_keep_one_frozen_arena_plan(
    monkeypatch, graph_factories, text_install_boundary, vision_install_boundary, cores, vision_cores,
):
    from rpu_backend.adapters.qwen3_5 import cores as profiles
    from rpu_backend.graph import GraphRuntimePolicy

    policy = GraphRuntimePolicy(execution_core_count=cores, graph_arena_count=4,
        lkn_max_batch_entries=999, lkn_kd_buf_mb=12, lkn_instr_buf_mb=48)
    model = SimpleNamespace()
    text = text_install_boundary
    text.install_qwen3_5_text_for_rpu(model, _config(9),
        execution_config={"model": {"num_cores": cores}}, _graph_runtime_policy=policy)
    state = model._rpu_qwen3_5
    state.vision_arena_admitted = True
    assert state.graph_cache.runtime_policy is state.prefill_graph.runtime_policy is policy
    assert graph_factories[0][1]["max_entries"] == 1
    with pytest.raises(ValueError, match="retained debug prefill"):
        text._qwen35_debug_prefill_cache(state)

    vision, visual, events = vision_install_boundary
    visual.config = _config(9).vision_config
    visual.blocks = visual.blocks + visual.blocks[:3]
    # Actual raw shapes/dtypes have independent real-HF boundary tests above.
    monkeypatch.setattr(profiles, "validate_dense_9b_vision_weights", lambda *a: None)
    monkeypatch.setattr(profiles, "validate_cold_weight_dtype", lambda *a, **kw: None)
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", lambda _: pytest.fail("lazy re-prepare"))
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "0")
    monkeypatch.setenv("RPU_FASTREPLAY_SKIP_SYNC", "invalid-after-cold-snapshot")
    options = vision._lazy_vision_graph_options(state)
    assert vision._lazy_vision_graph_options(state)["_graph_runtime_policy"] is options["_graph_runtime_policy"]
    assert (options["_graph_runtime_policy"] is policy) is (cores != 6)
    assert vision.install_qwen3_5_vision_for_rpu(visual,
        execution_config={"model": {"num_cores": cores}}, **options) == 23
    vp = visual._rpu_vision_graph_cache.runtime_policy
    assert vp is visual._rpu_vision_debug_graph.runtime_policy
    assert vp.execution_core_count == vision_cores and vp.graph_arena_count == 4
    assert (vp.lkn_max_batch_entries, vp.lkn_kd_buf_mb, vp.lkn_instr_buf_mb) == (999, 12, 48)
    assert vp.fast_replay_skip_sync is policy.fast_replay_skip_sync
    assert not vp.qwen35_legacy_27b_sdk_budget
    assert graph_factories[2][1]["max_entries"] == 1
    for block in visual.blocks:
        block._rpu_qwen3_5_vision_weights_converted = True
    assert vision.install_qwen3_5_vision_for_rpu(visual,
        execution_config={"model": {"num_cores": cores}}, **options) == 23
    assert events.count("create") == 1


def test_9b_late_opt_in_requires_reload_and_non9_lazy_path_is_unchanged(monkeypatch):
    from rpu_backend.adapters.qwen3_5.vision import _lazy_vision_graph_options

    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    with pytest.raises(RuntimeError, match="before to_rpu; reload"):
        _lazy_vision_graph_options(SimpleNamespace(vision_arena_admitted=False))
    assert _lazy_vision_graph_options(SimpleNamespace()) == {}


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("vision_opt_in", [False, True])
def test_9b_public_text_pool_keeps_vision_admission_separate(monkeypatch, cores, vision_opt_in):
    from rpu_backend.adapters import qwen3_5 as adapter
    from rpu_backend.adapters.qwen3_5 import cores as profiles, text, vision
    from rpu_backend.api._execution import bind_execution_session
    from rpu_backend.graph import GraphRuntimePolicy
    from rpu_backend.runtime import hw_attrs

    events, policies = [], []
    config = _config(9)
    inner = SimpleNamespace(config=config.text_config, embed_tokens=SimpleNamespace())
    inner.to = lambda device: events.append(("upload", device))
    # Keep only the output projection's metadata at the public boundary; its
    # actual large tensor geometry/dtype is checked by the real-HF tests above.
    output = SimpleNamespace(weight=SimpleNamespace(
        shape=(248320, 4096), detach=lambda: torch.zeros(1)))

    class Model:
        def __init__(self):
            self.config = config
            self.model = SimpleNamespace(language_model=inner)
            self.forward = object()
        def get_output_embeddings(self):
            return output
        def half(self):
            events.append("half")
            return self

    model = Model()
    owner = adapter.Qwen3_5Adapter.__new__(adapter.Qwen3_5Adapter)
    owner.model, owner._rpu_is_ready = model, False
    owner._execution_session = bind_execution_session(
        model, {"model": {"num_cores": cores}}, entry_point="9B public text pool",
        graph_mode="RETAINED_CACHE" if cores == 8 and not vision_opt_in else "BOUNDED_ONESHOT",
        supported=adapter.Qwen3_5Adapter._EXECUTION_SUPPORTED)
    owner._execution_session._bind_planner_owner(inner)
    model._rpu_execution = owner._execution_session.config
    if vision_opt_in:
        monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    else:
        monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    monkeypatch.setattr(adapter, "_claim_live_instance", lambda _: events.append("claim"))
    monkeypatch.setattr(profiles, "validate_dense_9b_vl_weights", lambda _: events.append("geometry"))
    monkeypatch.setattr(profiles, "validate_text_weight_geometry", lambda *a: None)
    monkeypatch.setattr(profiles, "validate_cold_weight_dtype", lambda *a, **kw: events.append("fp16"))
    for name in ("qwen3_5_set_execution_cores", "qwen3_5_get_execution_topology",
                 "qwen3_5_vision_set_execution_cores", "qwen3_5_vision_get_execution_topology",
                 "linear_with_accumulation"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *a: None, raising=False)

    def prepare(policy):
        policies.append(policy)
        events.append("prepare-pool")
        return True

    def install(language, root_config, max_seq_len, **kwargs):
        assert language is inner and root_config is config and max_seq_len == 128
        assert kwargs["_graph_runtime_policy"] is policies[0]
        assert kwargs["_public_prefill_graph_mode"] == owner._execution_session.graph_mode
        language._rpu_qwen3_5 = SimpleNamespace(
            graph_cache=SimpleNamespace(runtime_policy=policies[0]),
            execution_topology=profiles.core_topology(model._rpu_execution))
        events.append("install")
        return 23

    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", prepare)
    monkeypatch.setattr(adapter, "install_qwen3_5_text_for_rpu", install)
    monkeypatch.setattr(text, "_bind_qwen35_retirement", lambda *a: None)
    monkeypatch.setattr(adapter, "transform_linear_weight", lambda *a: SimpleNamespace(to=lambda _: torch.zeros(1)))
    monkeypatch.setattr(hw_attrs, "validate_postinstall", lambda _: None)
    monkeypatch.setattr(hw_attrs, "install_hw_attr_validator", lambda _: None)
    assert owner.to_rpu(max_seq_len=128) is model
    assert events == ["geometry", "claim", "prepare-pool", "half", "fp16", ("upload", "rpu"), "install"]
    policy, = policies
    assert policy.graph_arena_count == 4 and policy.execution_core_count == cores
    assert not policy.qwen35_legacy_27b_sdk_budget
    state = inner._rpu_qwen3_5
    assert state.vision_arena_admitted is vision_opt_in
    if vision_opt_in:
        assert vision._lazy_vision_graph_options(state)["_graph_runtime_policy"].graph_arena_count == 4
    else:
        with pytest.raises(RuntimeError, match="before to_rpu; reload"):
            vision._lazy_vision_graph_options(state)


@pytest.mark.parametrize('cores', [4, 6, 8])
@pytest.mark.parametrize('fault', ['no_device', 'late_pool', 'occupied'])
@pytest.mark.parametrize('vision_opt_in', [False, True])
def test_failed_preweight_pool_releases_real_python_live_claim(monkeypatch, cores, fault, vision_opt_in):
    from rpu_backend.adapters import qwen3_5 as adapter
    from rpu_backend.adapters.qwen3_5 import cores as profiles
    from rpu_backend.api import causal_lm
    from rpu_backend.api._execution import bind_execution_session
    from rpu_backend.graph import GraphRuntimePolicy

    monkeypatch.setattr(causal_lm, '_LIVE_REF', None)
    monkeypatch.setattr(causal_lm, '_LIVE_TERMINAL_REASON', None)
    config = _config(9)

    class Model:
        def __init__(self):
            self.config = config
            self.model = SimpleNamespace(language_model=SimpleNamespace(config=config.text_config))
            self.forward = object()

        def get_output_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(shape=(248320, 4096)))

        def half(self):
            pytest.fail('pool rejection must precede weight mutation')

    model = Model()
    owner = adapter.Qwen3_5Adapter.__new__(adapter.Qwen3_5Adapter)
    owner.model, owner._rpu_is_ready = model, False
    owner._execution_session = bind_execution_session(model, {'model': {'num_cores': cores}},
        entry_point='independent cold pool ownership',
        graph_mode='RETAINED_CACHE' if cores == 8 and not vision_opt_in else 'BOUNDED_ONESHOT',
        supported=adapter.Qwen3_5Adapter._EXECUTION_SUPPORTED)
    owner._execution_session._bind_planner_owner(model.model.language_model)
    model._rpu_execution = owner._execution_session.config
    if vision_opt_in:
        monkeypatch.setenv('QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED', '1')
    else:
        monkeypatch.delenv('QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED', raising=False)
    monkeypatch.setattr(adapter, '_claim_live_instance', causal_lm._claim_live_instance)
    monkeypatch.setattr(profiles, 'validate_dense_9b_vl_weights', lambda _: None)
    monkeypatch.setattr(profiles, 'validate_text_weight_geometry', lambda *a: None)
    for name in ('qwen3_5_set_execution_cores', 'qwen3_5_get_execution_topology',
                 'qwen3_5_vision_set_execution_cores', 'qwen3_5_vision_get_execution_topology',
                 'linear_with_accumulation'):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *a: None, raising=False)
    primary = RuntimeError('SDK pool frozen')

    def prepare(policy):
        assert causal_lm._LIVE_REF() is model
        if fault == 'late_pool':
            raise primary
        return False

    monkeypatch.setattr(GraphRuntimePolicy, 'prepare_arenas', prepare)
    occupied = Model() if fault == 'occupied' else None
    if occupied is not None:
        causal_lm._claim_live_instance(occupied)
    try:
        with pytest.raises((RuntimeError, adapter.RPUBackendError)) as caught:
            owner.to_rpu(max_seq_len=128)
        if fault == 'late_pool':
            assert caught.value is primary
        assert not getattr(model, '_rpu_swizzle_started', False)
        if occupied is not None:
            assert causal_lm._LIVE_REF() is occupied
            return
        # Keep the failed CPU model and its traceback alive. A second owner
        # must not be rejected as an already-live device model after no install.
        replacement = Model()
        causal_lm._claim_live_instance(replacement)
        assert causal_lm._LIVE_REF() is replacement
        causal_lm._release_live_instance(replacement)
    finally:
        causal_lm._release_live_instance(model)
        if occupied is not None:
            causal_lm._release_live_instance(occupied)


def test_real_hf_gdn_norm_neighbour_is_rejected_before_pool(monkeypatch):
    from transformers import Qwen3_5ForConditionalGeneration
    from rpu_backend.adapters.qwen3_5 import cores

    with torch.device('meta'):
        model = Qwen3_5ForConditionalGeneration(_config(9))
    # Device/dtype has a separate real-CPU boundary test. Here exercise actual
    # HF module shapes without allocating a 9B checkpoint.
    monkeypatch.setattr(cores, 'validate_cold_weight_dtype', lambda _: None)
    cores.validate_dense_9b_vl_weights(model)
    norm = model.model.language_model.layers[0].linear_attn.norm
    assert norm.weight.shape == (128,)
    norm.weight = torch.nn.Parameter(torch.empty(127, device='meta'))
    with pytest.raises(ValueError, match='shape'):
        cores.validate_dense_9b_vl_weights(model)


@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("where, name, value", [
    ("text", "intermediate_size", 3648), ("text", "hidden_size", 1056),
    ("text", "num_hidden_layers", 28), ("vision", "num_heads", 16),
    ("vision", "out_hidden_size", 2048), ("root", "tie_word_embeddings", False),
])
def test_08b_rejects_physical_checkpoint_and_neighbour_profiles(cores, where, name, value):
    config = _config(0.8)
    owner = {"text": config.text_config, "vision": config.vision_config, "root": config}[where]
    setattr(owner, name, value)
    with pytest.raises(ValueError):
        validate_core_profile(config, {"model": {"num_cores": cores}})
