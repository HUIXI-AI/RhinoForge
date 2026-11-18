"""Pi0.5 SigLIP Vision W4 (G32 RTN, quantized at install): host contracts only.

The recipe, its budgets (Vision on four lanes at 4/6/8) and the runner's cold
selectors; the adapter's four-lane install under budget 8 and the adapter and
installer refusals before any weight changes; the pgrp packing of the padded
projections at four lanes; and the native owner's declared op, its four-lane
check and the row-bias order. None of this is a board run or numerical
evidence for the W4 Vision route.
"""
import copy
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from examples.common import config as COMMON
from examples.runners.pi05 import optimized as runner
from rpu_backend.adapters import pi05, siglip
from rpu_backend.quant import precision_recipes
from rpu_backend.quant.int4_pgrp_pack import dequantize_int4_group_wise, quantize_int4_group_wise
from test_fmb_standalone_layout_rebuild import _definition
from test_hyvit2_four_lane_spm_layout import fmb_layout_prelude
from test_pi05_cold_install import _policy, isolated_process  # noqa: F401
from test_qwen3_vl_awq_packing import inverse_rpu_scale, inverse_rpu_weight


ROOT = Path(__file__).resolve().parents[1]
RECIPE = "w8a16_vision_w4a16_action_nvfp4"
CONFIG = ROOT / "examples/configs/pi05/h50" / f"{RECIPE}.toml"


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def selectors(monkeypatch):
    for name in ("RPU_PI05_SIGLIP_W4A16", "RPU_PI05_SIGLIP_W8A16", "RPU_PI05_SIGLIP_W8A16_SCOPE",
                 "RPU_PI05_SIGLIP_BATCH"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ------------------------------------------------------------- recipe and cores


def test_recipe_selects_vision_w4_on_the_action_nvfp4_checkpoint():
    family = precision_recipes.load(COMMON.PRECISION_RECIPES)["pi05"]
    item, base = family.recipes[RECIPE], family.recipes["w8a16_action_nvfp4"]
    assert item.selector == RECIPE and item.checkpoints == {"checkpoint": ("pi05", "w8a16_action_nvfp4")}
    assert item.spec("vision").key() == "w4a16-g32-rtn"
    assert item.limits == {"vision_on_install_quantization": "w4a16"}
    # Everything but Vision is the Action NVFP4 recipe it loads from.
    for probe in (("prefill",), ("prefill", "K"), ("action",), ("action", "K"), ("action", "AdaRMS")):
        assert item.spec(*probe) == base.spec(*probe), probe
    assert runner.recipe_route(RECIPE) == ("w4a16", False, True)
    assert runner.recipe_route("w8a16_action_nvfp4") == ("w4a16", False, False)


def test_example_takes_every_budget_with_vision_on_four_lanes(tmp_path):
    config = COMMON.load_config(CONFIG)
    assert config["rpu_execution"]["model"] == {"num_cores": 8}
    assert COMMON.core_support(config) == {4: None, 6: None, 8: None}
    text = CONFIG.read_text(encoding="utf-8")
    for cores in (4, 6):
        path = tmp_path / f"c{cores}.toml"
        path.write_text(text.replace("num_cores = 8", f"num_cores = {cores}"), encoding="utf-8")
        assert COMMON.load_config(path)["rpu_execution"]["model"] == {"num_cores": cores}
    # Omitted means eight: the TP8 paired Prefill/Action with serial four-lane Vision.
    omitted = copy.deepcopy(config)
    del omitted["rpu_execution"]
    execution, paired, chunk = runner.resolve_execution(runner.example_arguments(omitted))
    assert paired and chunk == 400 and execution["model"]["num_cores"] == 8
    # Eight takes T64-T160 like the other Action NVFP4 template; Vision stays single-image.
    longer = copy.deepcopy(config)
    longer["input"]["text_tokens"] = 160
    assert runner.resolve_execution(runner.example_arguments(longer))[1:] == (True, 464)
    for chunk, accepted in ((256, True), (768, False)):
        sized = copy.deepcopy(config)
        sized["rpu_execution"]["vision"] = {"chunk_size": chunk}
        if accepted:
            assert runner.resolve_execution(runner.example_arguments(sized))[1] is True
            continue
        with pytest.raises(ValueError, match=r"Vision chunk must be one of \('auto', 256\)"):
            runner.example_arguments(sized)


class _StoppedBeforeLoad(Exception):
    pass


def _cold_environment(monkeypatch, recipe, cores):
    config = COMMON.load_config(CONFIG)
    config["pi05"]["precision"] = recipe
    config["rpu_execution"]["model"]["num_cores"] = cores
    captured = []

    def bind(env):
        captured.append(dict(env))
        raise _StoppedBeforeLoad

    monkeypatch.setattr(runner, "bind_cold_environment", bind)
    with pytest.raises(_StoppedBeforeLoad):
        runner.run_profile(runner.example_arguments(config))
    return captured[0]


_EIGHT_LANE_VISION = ("RPU_PI05_VISION_MERGE_SEGMENT", "RPU_PI05_VISION_Q_RESIDENT",
                      "RPU_PI05_VISION_KV_SPM_ONLY", "RPU_PI05_VISION_OWNER_LN",
                      "RPU_PI05_VISION_FC_WEIGHT_OUTER", "RPU_PI05_SIGLIP_BATCH")


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_runner_selects_vision_w4_with_w8_scope_all_and_serial_vision(monkeypatch, cores):
    env = _cold_environment(monkeypatch, RECIPE, cores)
    assert (env["RPU_PI05_SIGLIP_W4A16"], env["RPU_PI05_SIGLIP_W8A16"],
            env["RPU_PI05_SIGLIP_W8A16_SCOPE"]) == ("1", "1", "all")
    # One camera at a time, without the eight-lane Vision routes.
    assert {env[name] for name in _EIGHT_LANE_VISION} == {"0"}
    base = _cold_environment(monkeypatch, "w8a16_action_nvfp4", cores)
    assert base["RPU_PI05_SIGLIP_W4A16"] == "0"
    if cores != 8:
        assert "RPU_PI05_SIGLIP_W8A16_SCOPE" not in base
        return
    # At eight the Prefill/Action routes are the Action NVFP4 template's; only Vision differs.
    assert base["RPU_PI05_SIGLIP_BATCH"] == base["RPU_PI05_VISION_OWNER_LN"] == "1"
    differ = {name for name in env.keys() | base.keys() if env.get(name) != base.get(name)}
    assert differ == {"RPU_PI05_SIGLIP_W4A16", "RPU_PI05_VISION_MERGE_SEGMENT",
                      "RPU_PI05_VISION_Q_RESIDENT", "RPU_PI05_VISION_KV_SPM_ONLY",
                      "RPU_PI05_VISION_OWNER_LN", "RPU_PI05_SIGLIP_BATCH"}, differ


def test_paired_profile_takes_vision_w4_with_action_nvfp4_only():
    for precision in ("fp16", "w8a16", "w4a16"):
        assert runner.profile_environment(precision)["RPU_PI05_SIGLIP_W4A16"] == "0"
    for precision in ("fp16", "w8a16"):
        with pytest.raises(ValueError, match="only with Action NVFP4"):
            runner.profile_environment(precision, vision_w4a16=True)
    with pytest.raises(ValueError, match="without the eight-lane Vision owner-LN"):
        runner.profile_environment("w4a16", vision_w4a16=True, vision_owner_ln=True)
    with pytest.raises(TypeError, match="vision_w4a16 must be bool"):
        runner.profile_environment("w4a16", vision_w4a16=1)
    env = runner.profile_environment("w4a16", vision_w4a16=True, prefill_w8a8=True, cameras=2,
                                     text_tokens=64)
    assert env["RPU_PI05_SIGLIP_W4A16"] == "1" and env["RPU_PI05_PREFILL_GATEUP_W8A8"] == "1"
    assert {env[name] for name in _EIGHT_LANE_VISION} == {"0"}


# ------------------------------------------------------------- adapter and installer refusals


def test_adapter_refuses_batched_vision_w4_or_without_w8_scope_all(selectors):
    selectors.setenv("RPU_PI05_SIGLIP_W4A16", "1")
    policy = _policy()
    adapter = pi05.Pi05Adapter(policy)
    claims = []
    selectors.setattr(pi05, "_preflight_pi05_cold_model",
                      lambda owner, **kw: (None,) * 10 + (True,))
    selectors.setattr(pi05, "_claim_live_instance", lambda owner: claims.append(owner))
    # Default eight cores with the batched Vision: refused before the live claim
    # or any weight change.
    with pytest.raises(pi05.RPUBackendError, match="one camera at a time"):
        adapter.to_rpu()
    assert claims == [] and not hasattr(policy, "_rpu_swizzle_started")

    four, six, eight = ({"model": {"num_cores": cores}} for cores in (4, 6, 8))
    quantized, fp16 = (None,) * 10 + (True,), (None,) * 10 + (False,)
    # Quantized VLM: W8 defaults on with scope all, so W4 is admitted on serial Vision.
    assert pi05._preflight_siglip_w4a16(four, quantized) is True
    assert pi05._preflight_siglip_w4a16(six, quantized) is True
    for execution in (eight, None):
        with pytest.raises(pi05.RPUBackendError, match="RPU_PI05_SIGLIP_BATCH=0"):
            pi05._preflight_siglip_w4a16(execution, quantized)
    selectors.setenv("RPU_PI05_SIGLIP_BATCH", "0")
    assert pi05._preflight_siglip_w4a16(eight, quantized) is True
    with pytest.raises(pi05.RPUBackendError, match="SCOPE=all"):
        pi05._preflight_siglip_w4a16(four, fp16)
    for name, value in (("RPU_PI05_SIGLIP_W8A16_SCOPE", "row"), ("RPU_PI05_SIGLIP_W8A16", "0")):
        selectors.setenv(name, value)
        with pytest.raises(pi05.RPUBackendError, match="SCOPE=all"):
            pi05._preflight_siglip_w4a16(eight, quantized)
    # Unselected: no check, and the cold model tuple is not read.
    selectors.setenv("RPU_PI05_SIGLIP_W4A16", "0")
    selectors.delenv("RPU_PI05_SIGLIP_BATCH")
    assert pi05._preflight_siglip_w4a16(eight, None) is False


class _StoppedAtEncoder(Exception):
    pass


@pytest.mark.parametrize("w4", (True, False))
def test_adapter_installs_vision_w4_on_four_lanes_under_budget_eight(selectors, w4):
    """The actual to_rpu at budget 8: Prefill/Action stay TP8, SigLIP W4 takes four lanes."""
    import rpu_backend.adapters.siglip as siglip_module
    from rpu_backend.adapters.pi05 import cores, w8a16, weights
    from rpu_backend.runtime import weights as runtime_weights

    selectors.setenv("RPU_PI05_SIGLIP_W4A16", str(int(w4)))
    selectors.setenv("RPU_PI05_SIGLIP_BATCH", "0")
    policy = _policy()
    adapter = pi05.Pi05Adapter(policy)
    decoders = NS(), NS()
    vision, projector = NS(), NS()
    events = []
    selectors.setattr(pi05, "_preflight_pi05_cold_model", lambda owner, **kw: (
        None, None, None, *decoders, vision, projector, 8, 8, {}, True))
    selectors.setattr(pi05, "_claim_live_instance", lambda owner: None)
    selectors.setattr(pi05, "_release_live_instance", lambda owner: None)
    selectors.setattr(torch.rpu, "set_caching_allocator", lambda enabled: None)
    selectors.setattr(w8a16, "replicate_pi05_decoder_kv_heads_for_attention_tp",
                      lambda model, effective_num_kv_heads, label: events.append(
                          (label, effective_num_kv_heads)) or 8)
    selectors.setattr(cores, "pad_decoder_mlp_inplace", lambda model, topology: None)
    selectors.setattr(runtime_weights, "convert_linear_weights_inplace",
                      lambda model, **kw: events.append(("linear", kw["attn_num_cores"],
                                                         kw["execution_core_count"])))
    selectors.setattr(weights, "swizzle_projector_inplace",
                      lambda owner, num_cores: events.append(("projector", num_cores)))

    def encoder(owner, *, w8a16_default, num_cores, w4a16):
        assert owner is vision and w8a16_default is True
        events.append(("encoder", num_cores, w4a16))
        raise _StoppedAtEncoder

    selectors.setattr(siglip_module, "_convert_siglip_weights_for_rpu", encoder)
    with pytest.raises(_StoppedAtEncoder):
        adapter.to_rpu()
    lanes = 4 if w4 else 8
    assert events == [("vlm", 8), ("expert", 8), ("linear", 8, 8), ("linear", 8, 8),
                      ("projector", lanes), ("encoder", lanes, w4)]


def test_budget_eight_keeps_the_private_cpu_camera_slab_only_for_four_lane_vision():
    """Serial cameras reach the four-lane SigLIP's native CPU ingress at every budget."""
    from rpu_backend.adapters.pi05 import runtime

    slab = torch.zeros(3, 3, 224, 224, dtype=torch.float16)
    images = [slab.narrow(0, index, 1) for index in range(3)]
    for index, image in enumerate(images):  # The batch preprocessor's slab marks.
        image._pi05_batch_slab, image._pi05_batch_index = slab, index

    def owner(cores, lanes):
        model = NS(_rpu_siglip_batch_n=1, _rpu_execution={"model": {"num_cores": cores}}, training=False)
        if lanes is not None:
            vision = NS(_rpu_vision_num_cores=lanes)
            model.paligemma_with_expert = NS(paligemma=NS(model=NS(vision_tower=NS(vision_model=vision))))
        return model

    for cores, lanes, admitted in ((4, None, True), (6, None, True), (8, 4, True),
                                   (8, 8, False), (8, None, False), (2, 4, False)):
        assert (runtime._eligible_serial_siglip_slab(owner(cores, lanes), images) is slab) is admitted, (cores, lanes)


def _fake_tower(dtype=torch.float16):
    def linear():
        return NS(weight=torch.zeros(2, 2, dtype=dtype))

    layer = NS(self_attn=NS(q_proj=linear(), k_proj=linear(), v_proj=linear(),
                            out_proj=linear(), embed_dim=1152, num_heads=16),
               mlp=NS(fc1=NS(weight=torch.zeros(2, 2, dtype=dtype), out_features=4304), fc2=linear()))
    config = NS(hidden_size=1152, intermediate_size=4304, num_attention_heads=16)
    return NS(config=config, encoder=NS(layers=[layer] * 27))


def test_installer_refuses_vision_w4_before_any_weight_changes(selectors):
    tower = _fake_tower()
    source = tower.encoder.layers[0].self_attn.q_proj.weight
    convert = siglip._convert_siglip_weights_for_rpu
    with pytest.raises(TypeError, match="w4a16 must be bool"):
        convert(tower, w8a16_default=True, num_cores=4, w4a16=1)
    with pytest.raises(ValueError, match="160 and 544 per core"):
        convert(tower, w8a16_default=True, num_cores=8, w4a16=True)
    with pytest.raises(ValueError, match="scope all"):
        convert(tower, w8a16_default=False, num_cores=4, w4a16=True)
    selectors.setenv("RPU_PI05_SIGLIP_W8A16_SCOPE", "fc2")
    with pytest.raises(ValueError, match="scope all"):
        convert(tower, w8a16_default=True, num_cores=4, w4a16=True)
    selectors.delenv("RPU_PI05_SIGLIP_W8A16_SCOPE")
    quantized = _fake_tower(torch.int8)
    with pytest.raises(ValueError, match="floating-point encoder weights"):
        convert(quantized, w8a16_default=True, num_cores=4, w4a16=True)
    small = _fake_tower()
    small.config.intermediate_size = 4096
    with pytest.raises(ValueError, match="exact Pi0.5 SigLIP27 geometry"):
        convert(small, w8a16_default=True, num_cores=4, w4a16=True)
    assert tower.encoder.layers[0].self_attn.q_proj.weight is source
    assert not hasattr(tower, "_rpu_siglip_weights_converted")
    # A converted tower keeps its weight format: switching W4 needs a reload.
    tower._rpu_siglip_weights_converted, tower._rpu_vision_num_cores = True, 4
    tower._siglip_w4a16 = False
    with pytest.raises(ValueError, match="Vision W4 is cold-only"):
        convert(tower, w8a16_default=True, num_cores=4, w4a16=True)


# ------------------------------------------------------------- packing


def _weight(n, k, salt):
    rows, cols = torch.arange(n)[:, None], torch.arange(k)[None, :]
    weight = ((rows * 7 + cols * 3 + salt) % 61 - 30).float() / 256
    weight[:, :32] *= 40  # one wide group: its scale must not spread to the next group
    return weight.half()


def test_padded_projections_pack_g32_at_four_lanes_and_detect_uniformly():
    # The converter pads head 72->80 and MLP 4304->4352 first, then quantizes,
    # so the G32 groups run over the physical K and the padding decodes to zero.
    def heads(weight):
        return torch.nn.functional.pad(weight.view(16, 72, -1), (0, 0, 0, 8)).reshape(1280, -1)

    raw = {"q": heads(_weight(1152, 1152, 1)), "k": heads(_weight(1152, 1152, 2)),
           "v": heads(_weight(1152, 1152, 3)),
           "o": torch.nn.functional.pad(_weight(1152, 1152, 4).view(1152, 16, 72), (0, 8)).reshape(1152, 1280),
           "fc1": torch.nn.functional.pad(_weight(4304, 1152, 5), (0, 0, 0, 48)),
           "fc2": torch.nn.functional.pad(_weight(1152, 4304, 6), (0, 48))}
    modules = {}
    for role, partition in (("q", 1), ("k", 1), ("v", 1), ("o", 0), ("fc1", 1), ("fc2", 0)):
        n, k = raw[role].shape
        module = modules[role] = torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
        siglip._install_siglip_linear_weight(
            module, raw[role], partition=partition, w8a16=True, num_cores=4, w4a16=True)
        weight, scale = module.weight, module.weight_scale
        codes, scale_gn, _ = quantize_int4_group_wise(raw[role], 32)
        assert weight.dtype == torch.uint8 and tuple(weight.shape) == (n, k // 2), role
        # The native set_weights check: aligned per-lane geometry and the exact G32 count.
        local_n, local_k = (n // 4, k) if partition == 1 else (n, k // 4)
        assert local_n % 16 == 0 and local_k % 64 == 0, role
        assert scale.dtype == torch.float16 and scale.dim() == 2 and scale.size(0) == 32, role
        assert scale.numel() == 4 * ((local_k // 32 + 3) // 4) * ((local_n + 63) // 64) * 256, role
        decoded = inverse_rpu_weight(weight, n, k, partition, 4)
        assert torch.equal(decoded, codes), role
        assert torch.equal(inverse_rpu_scale(scale, n, k, partition, 4), scale_gn.t()), role
        if role in ("q", "k", "v"):
            assert torch.count_nonzero(decoded.view(16, 80, -1)[:, 72:]) == 0, role
        elif role == "o":
            assert torch.count_nonzero(decoded.view(1152, 16, 80)[..., 72:]) == 0
        else:
            assert torch.count_nonzero(decoded[4304:] if role == "fc1" else decoded[:, 4304:]) == 0
        # Eight lanes split the row K into 160 / 544: no 64-aligned pgrp lane.
        if partition == 0:
            assert (k // 8) % 64 != 0, role

    def encoder(fc2):
        layer = NS(self_attn=NS(q_proj=modules["q"], k_proj=modules["k"], v_proj=modules["v"],
                                out_proj=modules["o"]), mlp=NS(fc1=modules["fc1"], fc2=fc2))
        return NS(layers=[layer, layer])

    assert siglip._detect_siglip_encoder_w4a16(encoder(modules["fc2"])) is True
    assert siglip._detect_and_validate_siglip_encoder_w8a16(encoder(modules["fc2"])) is False
    mixed = torch.nn.Linear(4352, 1152, bias=False, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="must cover all six projections"):
        siglip._detect_siglip_encoder_w4a16(encoder(mixed))


# ------------------------------------------------------------- LayerNorm beta fold


def test_folded_beta_reproduces_the_w4_linear_of_the_affine_layernorm():
    torch.manual_seed(0)
    x = torch.randn(8, 1152)
    gamma, beta = torch.rand(1152) + 0.5, torch.randn(1152).half()
    weight, bias = _weight(1280, 1152, 7), (torch.randn(1280) / 4).half()
    codes, scale, _ = quantize_int4_group_wise(weight, 32)
    weight_q = dequantize_int4_group_wise(codes, scale).float()
    expected = torch.nn.functional.layer_norm(x, (1152,), gamma, beta.float()) @ weight_q.T + bias.float()
    zero_beta = torch.nn.functional.layer_norm(x, (1152,), gamma, None) @ weight_q.T

    folded = siglip._fold_siglip_ln_beta(weight, bias, beta)
    assert folded.dtype == torch.float16 and folded.shape == bias.shape
    # One FP16 rounding of the folded bias; dropping beta is far outside it.
    atol = float(folded.float().abs().max()) * 2 ** -10
    torch.testing.assert_close(zero_beta + folded.float(), expected, atol=atol, rtol=0)
    assert float((zero_beta + bias.float() - expected).abs().max()) > 100 * atol


def _converter_tower():
    weights = {"qkvo": torch.zeros(1152, 1152, dtype=torch.float16),
               "fc1": torch.zeros(4304, 1152, dtype=torch.float16),
               "fc2": torch.zeros(1152, 4304, dtype=torch.float16)}

    def linear(weight, out):
        return NS(weight=weights[weight], bias=torch.arange(out, dtype=torch.float16) / 1024 + 1,
                  out_features=out)

    def layer(index):
        return NS(self_attn=NS(q_proj=linear("qkvo", 1152), k_proj=linear("qkvo", 1152),
                               v_proj=linear("qkvo", 1152), out_proj=linear("qkvo", 1152),
                               embed_dim=1152, num_heads=16),
                  mlp=NS(fc1=linear("fc1", 4304), fc2=linear("fc2", 1152)),
                  layer_norm1=NS(bias=torch.full((1152,), index + 1.0, dtype=torch.float16)),
                  layer_norm2=NS(bias=torch.full((1152,), -index - 1.0, dtype=torch.float16)))

    config = NS(hidden_size=1152, intermediate_size=4304, num_attention_heads=16)
    return NS(config=config, encoder=NS(layers=[layer(index) for index in range(27)]))


@pytest.mark.parametrize("w4", (True, False))
def test_w4_conversion_folds_each_beta_into_the_linears_its_layernorm_feeds(selectors, w4):
    """Q/K/V take LN1's beta and fc1 LN2's, over the padded weight and bias; the betas are zeroed."""
    tower, folds = _converter_tower(), []
    betas = [(layer.layer_norm1.bias, layer.layer_norm2.bias) for layer in tower.encoder.layers]
    selectors.setattr(siglip, "_install_siglip_linear_weight", lambda module, weight, **kw: None)

    def fold(weight, bias, beta):
        folds.append((tuple(weight.shape), bias.clone(), beta.clone()))
        return bias + 1

    selectors.setattr(siglip, "_fold_siglip_ln_beta", fold)
    siglip._convert_siglip_weights_for_rpu(tower, w8a16_default=True, num_cores=4, w4a16=w4)
    assert tower._siglip_ln_beta_folded is w4
    padded_head = torch.nn.functional.pad(
        (torch.arange(1152, dtype=torch.float16) / 1024 + 1).view(16, 72), (0, 8)).reshape(-1)
    padded_fc1 = torch.nn.functional.pad(torch.arange(4304, dtype=torch.float16) / 1024 + 1, (0, 48))
    if not w4:
        assert folds == []
        for layer, (ln1, ln2) in zip(tower.encoder.layers, betas):
            assert layer.layer_norm1.bias is ln1 and layer.layer_norm2.bias is ln2
            assert torch.equal(layer.self_attn.q_proj.bias, padded_head)
        return
    assert len(folds) == 27 * 4
    for index, (layer, (ln1, ln2)) in enumerate(zip(tower.encoder.layers, betas)):
        for shape, bias, beta in folds[4 * index:4 * index + 3]:
            assert shape == (1280, 1152) and torch.equal(bias, padded_head) and torch.equal(beta, ln1)
        shape, bias, beta = folds[4 * index + 3]
        assert shape == (4352, 1152) and torch.equal(bias, padded_fc1) and torch.equal(beta, ln2)
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            assert torch.equal(proj.bias, padded_head + 1)
        assert torch.equal(layer.mlp.fc1.bias, padded_fc1 + 1)
        for norm in (layer.layer_norm1, layer.layer_norm2):
            assert norm.bias.dtype == torch.float16 and torch.count_nonzero(norm.bias) == 0


def _siglip_layout_program():
    source = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text(encoding="utf-8")
    code = '#include "fused/rpu_siglip_w8a8_contract.h"\n' + fmb_layout_prelude()
    declare = _definition(source, "std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override")
    code += r'''
struct Probe {
 int64_t cores=4,image_batch_count_=1,reduced_w8_projection_mask_=0;
 bool q_resident=false,qkv_w8a8_=false,mlp_w8a8_=false,pi05_vision_ln_stream_opt_in_=false;
 int64_t hidden_size()const{return 1152;}int64_t num_q_heads()const{return 16;}
 int64_t head_dim()const{return 80;}int64_t intermediate_size()const{return 4352;}
 int64_t num_cores()const{return cores;}int64_t mlp_tp()const{return cores;}
 int64_t num_layers()const{return 27;}
 bool use_pi05_vision_q_resident(int64_t rows,AttentionExecutionPolicy p)const{
  return q_resident&&p==AttentionExecutionPolicy::SPM_KV_BY_MHA&&rows==image_batch_count_*256;}
'''
    code += _definition(source, "bool ln_beta_folded() const") + "\n"
    code += _definition(source, "int64_t ln_stream_elems() const") + "\n"
    code += declare.replace(" override {", " {", 1) + "\n};\n}\n"
    code += r'''
using namespace v3;
static bool has(const std::vector<BufferDecl>& d,const char* n){
 for(const auto& b:d)if(std::string(b.name)==n)return true;return false;}
int main(){
 Probe m;LayoutContext lc;lc.chunk_size=256;
 for(auto [cores,mask]:{std::pair<int64_t,int64_t>{8,0},{4,0},{4,0x3f},{4,0xfc0}}){
  m.cores=cores;m.reduced_w8_projection_mask_=mask;const auto d=m.declare_buffers(lc);
  int64_t per_layer=0,global=0;
  for(const auto& b:d){
   if(b.storage==StorageClass::PersistentPerLayer)per_layer+=b.size;
   if(b.storage==StorageClass::Persistent)global+=b.size;
  }
  const bool folded=mask==0xfc0;
  assert(has(d,"ln1_beta")==!folded&&has(d,"ln2_beta")==!folded&&has(d,"ln_beta_zero")==folded);
  std::cout<<cores<<" "<<mask<<" "<<per_layer<<" "<<global<<" "<<27*per_layer+global<<"\n";
 }
 // Four-lane packed N=2/3 (512/768 rows): both the initial DDR_KV chunk-domain
 // probe and the physical SPM_KV_BY_MHA+Q_RESIDENT layout must fit alongside
 // SigLIP + Gemma + Denoise super-persistent allocations.
 m.cores=4;m.reduced_w8_projection_mask_=0;m.q_resident=true;
 for(int64_t images:{2,3}){
  m.image_batch_count_=images;
  for(auto pol:{AttentionExecutionPolicy::DDR_KV,AttentionExecutionPolicy::SPM_KV_BY_MHA}){
   LayoutContext plc;plc.chunk_size=images*256;plc.attention_policy=pol;
   const auto pd=m.declare_buffers(plc);
   std::vector<SpmAllocator::AllocRequest> reqs;
   for(const auto& b:pd)if(b.storage==StorageClass::Temp&&!b.alias_of)
    reqs.push_back({b.size,b.phase_start,b.phase_end,static_cast<int>(b.scope)});
   const auto plan=SpmAllocator::plan_temporary_aliased(reqs);
   assert(plan.peak_bytes+571904+153856+65536<=SpmAllocator::SPM_USABLE);
  }
 }
}
'''
    return code


def test_four_lane_w4_vision_keeps_persistent_spm_inside_the_eight_lane_tower(tmp_path):
    """Folded betas drop 27 x 2 per-layer slots for one zero vector.

    The TP8 Pi0.5 prefill planner reserves the Vision tower's persistent SPM.
    At four lanes the per-core Q/K/V and fc1 bias slices grow (20,992 against
    18,432 B per layer), which left no TP8 prefill chunk for W4 Vision on the
    board; packed W4 sheds both betas and lands below the eight-lane tower.
    """
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    cpp, binary = tmp_path / "siglip_layout.cpp", tmp_path / "siglip_layout"
    cpp.write_text(_siglip_layout_program())
    result = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src"),
                             str(cpp), "-o", str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    rows = {(int(c), int(m)): (int(p), int(g), int(t))
            for c, m, p, g, t in (line.split() for line in result.stdout.splitlines())}
    assert rows[8, 0] == (18_432, 5_120, 502_784)
    assert rows[4, 0] == rows[4, 0x3F] == (20_992, 5_120, 571_904)
    assert rows[4, 0xFC0] == (15_872, 7_680, 436_224)
    assert rows[4, 0xFC0][2] < rows[8, 0][2]


# ------------------------------------------------------------- native owner and installer


def test_native_owner_declares_w4_on_four_lanes_and_adds_row_bias_before_the_reduction():
    source = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text(encoding="utf-8")
    check = source[source.index("static void check_siglip_w4a16_scale_lists("):]
    check = check[:check.index("\n}\n")]
    # Column q/k/v/fc1 and row o/fc2 at the actual lanes, with the G32 payload size.
    assert "n % (col ? 16 * cores : 16) == 0" in check and "k % (col ? 64 : 64 * cores) == 0" in check
    assert "s.size(0) == 32" in check and "s.numel() == expected" in check
    for role, col in (("q", "true"), ("k", "true"), ("v", "true"), ("o", "false"),
                      ("fc1", "true"), ("fc2", "false")):
        assert re.search(rf"check_projection\({role}_w,\s+{role}_ws,\s+{col},", check), role
    body = source[source.index("if (packed_w4) {"):]
    assert re.match(r"if \(packed_w4\) \{(\s+//[^\n]*)+\s+TORCH_CHECK\(num_cores\(\) == 4 && N == 27", body)
    op = source[source.index("void rpu_siglip_set_weights_w4a16("):]
    assert "/*packed_w4=*/true);" in op[:op.index("\n}\n")]
    # The packed W4 row Linear refuses in-kernel bias: launch without it, add the
    # core-0-only row bias to every partial, then reduce.
    for flag, bias in (("w4_o", "o_bias"), ("w4_fc2", "fc2_bias")):
        launch = source.index(f'{flag} ? 0 : layer_addr(layer_idx, 0, "{bias}")')
        add = source.index(f'if ({flag}) add_row_bias_partial(layer_addr(layer_idx, 0, "{bias}")')
        reduce = source.index("rpu_launch_all_reduce_sum_residual_kernel(", add)
        assert launch < add < reduce, flag
    # CPU single-image ingress admits the four-lane W4 owner; eight-core opt-ins do not.
    assert "(pi05_vision_supported_owner() || pi05_vision4_w4_owner())" in source
    assert source.count("pi05_vision4_w4_owner()") == 2
    for path in ("src/core/rpu_kernel_decls.h", "src/core/rpu_dispatch_registrations.inc"):
        assert "siglip_set_weights_w4a16" in (ROOT / path).read_text(encoding="utf-8"), path
    installer = (ROOT / "python/rpu_backend/adapters/siglip.py").read_text(encoding="utf-8")
    assert re.search(r"if siglip_w4a16:\n\s+torch\.ops\.rpu\.siglip_set_weights_w4a16\(", installer)
    # The Graph signature separates FP16 (0), W8 (1) and packed W4 (2).
    assert "_siglip_sig_w8a16 = 2 if siglip_w4a16 else int(siglip_w8a16)" in installer
