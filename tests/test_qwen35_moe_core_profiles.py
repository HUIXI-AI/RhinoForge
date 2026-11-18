"""Exercise cold core selection through real MoE cache and weight consumers."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api.errors import UnsupportedModelError

from test_qwen35_moe_python_integration import _exact_config


def config():
    from transformers import Qwen3_5MoeConfig
    return Qwen3_5MoeConfig.from_dict(_exact_config())


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_fixed_profile_reaches_real_cache_geometry_and_reset(monkeypatch, cores):
    from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
    from rpu_backend.adapters.qwen3_5_moe.cores import core_topology
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache

    execution = {"model": {"num_cores": cores}}
    attention = 8 if cores == 8 else 4
    cfg = config()
    Qwen3_5MoeAdapter.preflight_execution(cfg, execution)
    topology = core_topology(execution)
    state = SimpleNamespace(handle=17, execution_topology=topology)
    model = SimpleNamespace(config=cfg, _rpu_execution=execution,
                            _rpu_qwen3_5_moe=state)
    calls = []
    def native(handle):
        calls.append(handle)
        return (cores, attention, cores, attention, 8)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_get_execution_topology", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_get_execution_topology_v2",
        lambda _: (2, cores, attention, attention, attention, cores, cores, cores, attention, 8),
        raising=False)
    cache = Qwen3_5MoeCache.from_model(model, max_seq_len=64, device="cpu")
    assert calls == [17]
    assert cache.execution_topology == (cores, attention, cores, attention, 8)
    assert cache.num_kv_heads == cache.attn_tp == cache.gdn_num_cores == attention
    assert cache.physical_kv_cores == cache.nKVHeadChunk == 8
    assert cache.k_caches[3].shape[4] == 8
    assert cache.k_caches[0].numel() == 0
    assert cache.conv_states[0].shape == (attention, 4, 8192 // attention)
    assert cache.recurrent_states[0].shape == (32, 128, 128)
    arrays = [cache.k_caches[3], cache.v_caches[3], cache.conv_states[0], cache.recurrent_states[0]]
    pointers = [a.data_ptr() for a in arrays]
    for a in arrays:
        a.fill_(3)
    cache.position = 5
    cache.reset()
    assert cache.position == 0
    assert all(torch.count_nonzero(a) == 0 for a in arrays)
    assert pointers == [a.data_ptr() for a in arrays]


@pytest.mark.parametrize("cores", [3, 5, True, 4.0])
def test_unsupported_core_choice_fails_before_any_cache_allocation(monkeypatch, cores):
    from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache

    cfg = config()
    def no_allocate(*args, **kwargs):
        pytest.fail("invalid profile reached allocation")
    monkeypatch.setattr(torch, "zeros", no_allocate)
    with pytest.raises(UnsupportedModelError, match="cores"):
        Qwen3_5MoeAdapter.preflight_execution(cfg, {"model": {"num_cores": cores}})
    with pytest.raises(ValueError, match="cores"):
        Qwen3_5MoeCache.from_config(cfg, 64, device="cpu", num_cores=cores)


@pytest.mark.parametrize("drift", ["config", "native", "physical_kv"])
def test_cache_refuses_stale_installed_topology_before_allocation(monkeypatch, drift):
    from rpu_backend.adapters.qwen3_5_moe.cores import core_topology
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache

    state = SimpleNamespace(handle=17, execution_topology=core_topology({"model": {"num_cores": 4}}))
    model = SimpleNamespace(config=config(), _rpu_qwen3_5_moe=state,
                            _rpu_execution={"model": {"num_cores": 8 if drift == "config" else 4}})
    native = (8, 8, 8, 8, 8) if drift == "native" else (4, 4, 4, 4, 4)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_get_execution_topology", lambda _: native, raising=False)
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: pytest.fail("topology drift allocated cache"))
    with pytest.raises((ValueError, RuntimeError), match="(layout|topology)"):
        Qwen3_5MoeCache.from_model(model, 64, device="cpu")


@pytest.mark.parametrize("missing", [
    "qwen3_5_moe_set_execution_cores", "qwen3_5_moe_get_execution_topology",
    "qwen3_5_moe_get_execution_topology_v2",
])
def test_missing_native_topology_api_fails_before_text_weight_mutation(monkeypatch, missing):
    from rpu_backend.adapters.qwen3_5_moe.text import install_qwen3_5_moe_text_for_rpu

    model = SimpleNamespace()
    for name in ("qwen3_5_moe_set_execution_cores", "qwen3_5_moe_get_execution_topology",
                 "qwen3_5_moe_get_execution_topology_v2"):
        monkeypatch.setattr(torch.ops.rpu, name,
                            None if name == missing else lambda *a: (), raising=False)
    with pytest.raises(RuntimeError, match="binary lacks core topology op"):
        install_qwen3_5_moe_text_for_rpu(model, config(), max_seq_len=64,
                                        execution_config={"model": {"num_cores": 4}})
    assert not hasattr(model, "_rpu_qwen3_5_moe_text_install_started")


@pytest.mark.parametrize("axis", [0, 3, 4, 5, 6, 7, 8, 9])
def test_tp6_full_native_owner_drift_fails_before_cache_allocation(monkeypatch, axis):
    from rpu_backend.adapters.qwen3_5_moe.cores import core_topology
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache

    state = SimpleNamespace(handle=17, execution_topology=core_topology({"model": {"num_cores": 6}}))
    model = SimpleNamespace(config=config(), _rpu_qwen3_5_moe=state,
                            _rpu_execution={"model": {"num_cores": 6}})
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_get_execution_topology",
                        lambda _: (6, 4, 6, 4, 8), raising=False)
    full = [2, 6, 4, 4, 4, 6, 6, 6, 4, 8]
    full[axis] = 1 if axis == 0 else (6 if full[axis] == 4 else 4)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_get_execution_topology_v2",
                        lambda _: full, raising=False)
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: pytest.fail("owner drift reached allocation"))
    with pytest.raises(RuntimeError, match="component topology mismatch"):
        Qwen3_5MoeCache.from_model(model, 64, device="cpu")


def test_specialized_skipped_linear_stays_raw_with_reduced_conversion():
    from rpu_backend.runtime.weights import convert_linear_weights_inplace, transform_linear_weight

    module = torch.nn.Module()
    module.in_proj_z = torch.nn.Linear(256, 128, bias=False, dtype=torch.float16)
    module.o_proj = torch.nn.Linear(256, 128, bias=False, dtype=torch.float16)
    raw = module.in_proj_z.weight.detach().clone()
    expected = transform_linear_weight(module.o_proj.weight.detach().clone(), 0, 4)
    convert_linear_weights_inplace(module, attn_num_cores=4, mlp_num_cores=4,
                                   lm_head_num_cores=4, execution_core_count=4,
                                   skip_names={"in_proj_z"})
    assert torch.equal(module.in_proj_z.weight, raw)
    assert torch.equal(module.o_proj.weight, expected)
    assert module.o_proj._rpu_linear_num_cores == 4

    unknown = torch.nn.Module()
    unknown.unclaimed = torch.nn.Linear(256, 128, bias=False, dtype=torch.float16)
    raw = unknown.unclaimed.weight.detach().clone()
    with pytest.raises(ValueError, match="unknown Linear role"):
        convert_linear_weights_inplace(unknown, attn_num_cores=4, mlp_num_cores=4,
                                       lm_head_num_cores=4, execution_core_count=4)
    assert torch.equal(unknown.unclaimed.weight, raw)


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("keep", [0, 1])
@pytest.mark.parametrize("acc32", [False, True])
def test_public_forward_uses_installed_head_core_layout(monkeypatch, cores, keep, acc32):
    from rpu_backend.adapters import qwen3_5_moe as moe
    from rpu_backend.adapters.qwen3_5_moe.cores import core_topology
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache
    from rpu_backend.runtime import hw_attrs

    generator = torch.Generator().manual_seed(58)
    output = torch.randn(1, 3, 16, generator=generator).half()
    weight = torch.randn(64, 16, generator=generator).half()
    inner = SimpleNamespace(_rpu_qwen3_5_moe=SimpleNamespace(
        execution_topology=core_topology({"model": {"num_cores": cores}}),
        linear_acc32=acc32))
    fusion = SimpleNamespace(language_model=inner, visual=object(), rope_deltas=None)
    model = SimpleNamespace(model=fusion, _lm_head_w_rpu_keepalive=weight)
    cache = object.__new__(Qwen3_5MoeCache)
    cache.position = 1  # Keep the independent text-only reset bridge out of this head test.
    monkeypatch.setattr(hw_attrs, "mark_first_forward_done", lambda _: None)
    monkeypatch.setattr(moe, "_preflight_qwen3_5_moe_text", lambda *a, **k: object())
    monkeypatch.setattr(moe, "run_qwen3_5_moe_text", lambda *a, **k: output)
    original_to = torch.Tensor.to
    def host_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", host_to)
    calls = []
    linear = torch.nn.functional.linear
    def explicit_head(flat, actual_weight, bias, partition, count, enabled):
        calls.append((partition, count, tuple(flat.shape), enabled))
        assert actual_weight is weight and bias is None
        return linear(flat, actual_weight)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", explicit_head, raising=False)
    result = moe._rpu_qwen3_5_moe_forward.__wrapped__(
        model, inputs_embeds=output, past_key_values=cache, logits_to_keep=keep)
    expected_input = output[:, -keep:] if keep else output
    torch.testing.assert_close(result.logits, linear(expected_input, weight).float(), rtol=0, atol=0)
    assert result.past_key_values is cache
    assert calls == [(1, 4 if cores != 8 else 8, (1 if keep else 3, 16), acc32)]


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_public_example_core_choice_matches_its_registered_profile(monkeypatch, tmp_path, cores):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("moe_core_example", root / "examples/common/config.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    path = root / "examples/configs/qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml"
    baseline = common.load_config(path)
    selected = tmp_path / "selected.toml"
    selected.write_text(path.read_text().replace("num_cores = 8", f"num_cores = {cores}"))
    config = common.load_config(selected)
    assert config["rpu_execution"]["model"]["num_cores"] == cores
    assert config["example"]["profile_id"] == baseline["example"]["profile_id"]
    selected.write_text(path.read_text().replace("num_cores = 8", "num_cores = 5"))
    with pytest.raises(common.ConfigError, match="model.num_cores"):
        common.load_config(selected)


@pytest.mark.parametrize("enabled", [False, True])
def test_moe_text_accumulation_is_public_cold_and_immutable(monkeypatch, enabled):
    from rpu_backend.adapters import qwen3_5_moe as moe
    from rpu_backend.api._execution import normalize_rpu_execution

    selected = normalize_rpu_execution(
        {"prefill": {"linear_acc32": enabled}}, entry_point="MoE text",
        supported=moe.Qwen3_5MoeAdapter._EXECUTION_SUPPORTED)
    assert selected["prefill"]["linear_acc32"] is enabled
    moe.Qwen3_5MoeAdapter.preflight_execution(config(), selected)
    instance = SimpleNamespace(model=SimpleNamespace(config=config()),
        preflight_execution=moe.Qwen3_5MoeAdapter.preflight_execution)
    monkeypatch.setattr(moe, "_qwen3_5_moe_cold_config_only", lambda _: False)
    with pytest.raises(ValueError, match="immutable after (RPU|device) installation"):
        moe._validate_qwen3_5_moe_execution_reconfigure(instance, selected)
