"""CPU regressions for the opt-in Pi0.5 W8/Action-W4 integration."""
import json
import importlib
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from rpu_backend.adapters.pi05 import runtime
from rpu_backend.adapters.pi05.w4pack import int4_child_names, pack_int4_projections_inplace
from rpu_backend.quant.convert_pi05 import convert_checkpoint, PI05_EXPERT_ANCHOR, PI05_VLM_ANCHOR

ROOT = Path(__file__).resolve().parents[1]
RUNNER = vars(importlib.import_module("examples.runners.pi05.optimized"))


@pytest.mark.parametrize("precision,fmt", [("w4a16", "nvfp4"), ("int4_g128", "int4_g128"), ("legacy_int4", "legacy_int4")])
def test_action_checkpoint_conversion_preserves_w8_vlm_and_kv(tmp_path, precision, fmt):
    src = tmp_path / "fp16"
    src.mkdir()
    (src / "config.json").write_text("{}")
    tensor = torch.linspace(-1, 1, 256).repeat(256, 1).half()
    names = ["model" + anchor + "0.self_attn." + proj + ".weight"
             for anchor in (PI05_EXPERT_ANCHOR, PI05_VLM_ANCHOR)
             for proj in ("q_proj", "k_proj", "v_proj")]
    save_file({name: tensor.clone() for name in names}, src / "model.safetensors")
    dst = tmp_path / "action-w4"
    # Omitting the format must exercise the public NVFP4 default.
    options = {} if fmt == "nvfp4" else {"action_w4_format": fmt}
    convert_checkpoint(src, dst, action_w4=True, **options)
    cfg = json.loads((dst / "rpu_quant_config.json").read_text())
    RUNNER["validate_checkpoint"](cfg, precision)
    if fmt != "legacy_int4":
        assert cfg["qaxis"] == 1
    if fmt == "nvfp4":
        assert cfg["kernel"] == "wnvfp4a16"
        assert cfg["nvfp4_abi"] == "striped_v2"
        assert "accumulation selected by cold linear_acc32 config" in cfg["note"]
    assert int4_child_names(cfg, component="vlm") == set()
    assert int4_child_names(cfg, component="expert") == {"q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    with pytest.raises(ValueError, match="explicit component"):
        int4_child_names(cfg)
    values = load_file(dst / "model.safetensors")
    for name in names:
        if PI05_EXPERT_ANCHOR in name and name.endswith("q_proj.weight"):
            assert values[name].dtype == (torch.uint8 if fmt == "nvfp4" else torch.int8)
            assert values[name].abs().max() <= (15 if fmt == "nvfp4" else 8)
            if fmt == "int4_g128":
                assert values[name.replace(".weight", ".weight_scale")].shape == (2, 256)
        else:
            assert values[name].abs().max() > 8
    assert not (dst / "model_remapped.safetensors").exists()


def test_gs128_packing_is_idempotent_and_leaves_kv_unchanged():
    attn = torch.nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        module = torch.nn.Linear(1024, 256, bias=False)
        module.weight = torch.nn.Parameter(torch.randint(-8, 8, (256, 1024), dtype=torch.int8), requires_grad=False)
        module.register_buffer("weight_scale", torch.ones(8, 256, dtype=torch.float16))
        module._pi05_int4_group_size = 128
        setattr(attn, name, module)
    decoder = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
    kv = {name: getattr(attn, name).weight.clone() for name in ("k_proj", "v_proj")}
    assert pack_int4_projections_inplace(decoder, {"q_proj", "o_proj"}, attn_num_cores=8, group_size=128) == 2
    packed = attn.q_proj.weight.clone()
    assert packed.dtype == torch.uint8 and packed.shape == (256, 512)
    assert attn.q_proj.weight_scale.size(0) == 128
    assert pack_int4_projections_inplace(decoder, {"q_proj", "o_proj"}, attn_num_cores=8, group_size=128) == 0
    assert torch.equal(packed, attn.q_proj.weight)
    for name, value in kv.items():
        assert torch.equal(value, getattr(attn, name).weight)


def test_profiles_reject_wrong_precision_scope():
    cfg = {"method": "w4a16", "mixed_int8_suffixes": ["k_proj.weight", "v_proj.weight"]}
    with pytest.raises(ValueError, match="Action-only"):
        RUNNER["validate_checkpoint"](cfg, "legacy_int4")
    with pytest.raises(ValueError, match="requested w8a16"):
        RUNNER["validate_checkpoint"](cfg, "w8a16")
    for precision in ("w8a16", "w4a16", "int4_g128", "legacy_int4"):
        env = RUNNER["profile_environment"](precision)
        # Both optimized KV1 routes require M50 to occupy 64 cache rows.
        assert env["RPU_PI05_KVINSERT_PAD16"] == "1"
        assert env["RPU_PI05_DENOISE_W4_GEGLU_N64"] == str(int(precision in ("int4_g128", "legacy_int4")))
        assert env["RPU_PI05_DENOISE_GEGLU_M50"] == str(int(precision == "w8a16"))
        assert env["RPU_PI05_DENOISE_ELIDE_UNUSED_DMA"] == "0"
        assert env["RPU_PI05_DENOISE_NVFP4_GEGLU_M50"] == "0"
        assert env["RPU_PI05_DENOISE_W4_FASTPATH"] == str(int(precision != "w8a16"))
        for flag in ("XOR3_GATED_ATTN", "XOR3_GATED_MLP", "KV1_DIRECT_CACHE", "KV1_PAIR_OWNER"):
            assert env["RPU_PI05_DENOISE_" + flag] == "1"


@pytest.mark.parametrize("profile,expected", [
    ((True, "w4a16", True, 3, 160), 1),
    ((True, "w4a16", False, 3, 160), 24),
    ((True, "w4a16", True, 2, 160), 24),
    ((False, "w4a16", True, 3, 160), 24),
])
def test_cpu_interop_threads_are_bounded_only_for_validated_t160_profile(profile, expected):
    paired, precision, prefill_w8a8, cameras, text_tokens = profile
    assert RUNNER["_cpu_interop_threads"](
        paired=paired, precision=precision, prefill_w8a8=prefill_w8a8,
        cameras=cameras, text_tokens=text_tokens) == expected


def test_fp16_profile_keeps_weights_unquantized_and_rejects_quantized_checkpoint():
    env = RUNNER["profile_environment"]("fp16")
    for flag in ("SIGLIP_W8A16", "ADARMS_DENSE_W8A16", "PREFILL_GATEUP_W8A8",
                 "VISION_OWNER_LN", "DENOISE_W4_FASTPATH", "DENOISE_NVFP4_GEGLU_M50"):
        assert env["RPU_PI05_" + flag] == "0"
    for flag in ("KVINSERT_PAD16", "DENOISE_UNROLL", "PREFILL_MASK_FP16",
                 "FUSED_DENOISE", "SIGLIP_BATCH", "VISION_KV_SPM_ONLY", "PREFILL_OWNER_NORM",
                 "PREFILL_KV_ONLY", "PREFILL_GATEUP_PAIR", "DENOISE_KV1_PAIR_OWNER", "DENOISE_GEGLU_M50"):
        assert env["RPU_PI05_" + flag] == "1"
    RUNNER["validate_checkpoint"]({}, "fp16")
    for config in ({"method": "w8a16"}, {"method": "nvfp4a16"}, {"mode": "unknown"}):
        with pytest.raises(ValueError, match="unquantized checkpoint"):
            RUNNER["validate_checkpoint"](config, "fp16")


def test_research_routes_are_explicit_and_precision_scoped():
    profile = RUNNER["profile_environment"]
    # Old experiment callers bind these separately; their one-argument call
    # must not overwrite an existing external choice.
    legacy = profile("w4a16")
    assert "RPU_PI05_PREFILL_GATEUP_W8A8" not in legacy
    assert "RPU_PI05_VISION_OWNER_LN" not in legacy
    off = profile("w4a16", prefill_w8a8=False, vision_owner_ln=False)
    on = profile("w4a16", prefill_w8a8=True, vision_owner_ln=True)
    for flag in ("PREFILL_GATEUP_W8A8", "VISION_OWNER_LN"):
        assert off["RPU_PI05_" + flag] == "0"
        assert on["RPU_PI05_" + flag] == "1"
    for precision in ("fp16", "w8a16", "int4_g128", "legacy_int4"):
        with pytest.raises(ValueError, match="only with Action NVFP4"):
            profile(precision, prefill_w8a8=True)
    assert profile("fp16", vision_owner_ln=True)["RPU_PI05_VISION_OWNER_LN"] == "1"
    with pytest.raises(TypeError, match="must be bool"):
        profile("w4a16", prefill_w8a8="false")


def test_three_camera_entry_rejects_missing_or_masked_cameras_and_other_prefixes():
    cameras = {"observation.images." + name: object() for name in ("base", "left", "right")}
    batch = {key: torch.zeros(1, 3, 8, 8) for key in cameras}
    batch["observation.language.tokens"] = torch.zeros(1, 32, dtype=torch.long)
    validate = RUNNER["validate_three_camera_batch"]
    validate(batch, cameras)
    for count in (1, 2):
        reduced = {key: batch[key] for key in list(cameras)[:count]}
        reduced["observation.language.tokens"] = batch["observation.language.tokens"]
        with pytest.raises(ValueError, match="all three configured cameras"):
            validate(reduced, cameras)
        with pytest.raises(ValueError, match="all three configured cameras"):
            validate(reduced, list(cameras)[:count])
    masked = {**batch, next(iter(cameras)) + "_padding_mask": torch.zeros(1, dtype=torch.bool)}
    with pytest.raises(ValueError, match=r"present \(true\)"):
        validate(masked, cameras)
    with pytest.raises(ValueError, match="32 language tokens"):
        validate({**batch, "observation.language.tokens": torch.zeros(1, 48)}, cameras)


@pytest.mark.parametrize("precision", ["fp16", "w8a16", "w4a16"])
def test_two_camera_profile_transfers_optimizations_without_changing_precision(precision):
    profile = RUNNER["profile_environment"]
    env = profile(precision, cameras=2)
    for flag in ("KVINSERT_PAD16", "DENOISE_UNROLL", "PREFILL_MASK_FP16", "FUSED_DENOISE", "SIGLIP_BATCH"):
        assert env["RPU_PI05_" + flag] == "1"
    for flag in RUNNER["_ON"]:
        assert env["RPU_PI05_" + flag] == "1"
    for flag in ("PREFILL_GATEUP_W8A8", "VISION_OWNER_LN"):
        assert env["RPU_PI05_" + flag] == "0"
    assert env["RPU_PI05_DENOISE_GEGLU_M50"] == str(int(precision in ("fp16", "w8a16")))
    assert env["RPU_PI05_DENOISE_W4_FASTPATH"] == str(int(precision == "w4a16"))
    assert env["RPU_PI05_DENOISE_NVFP4_GEGLU_M50"] == "0"
    for flag in ("SIGLIP_W8A16", "ADARMS_DENSE_W8A16"):
        assert env["RPU_PI05_" + flag] == str(int(precision != "fp16"))
    if precision == "w4a16":
        on = profile(precision, cameras=2, prefill_w8a8=True)
        assert on == {**env, "RPU_PI05_PREFILL_GATEUP_W8A8": "1"}
    else:
        with pytest.raises(ValueError, match="only with Action NVFP4"):
            profile(precision, cameras=2, prefill_w8a8=True)
    assert profile(precision, cameras=2, vision_owner_ln=True)["RPU_PI05_VISION_OWNER_LN"] == "1"


def test_prefill_a8_chunk_is_exact_and_other_profiles_keep_explicit_chunks():
    resolve = RUNNER["resolve_prefill_chunk_size"]
    for cameras, expected in ((2, 272), (3, 400)):
        for requested in (None, expected):
            assert resolve(cameras=cameras, prefill_w8a8=True, requested=requested) == expected
        for requested in (256, 288, 400 if cameras == 2 else 272):
            with pytest.raises(ValueError, match=f"requires exactly --prefill-chunk-size {expected}"):
                resolve(cameras=cameras, prefill_w8a8=True, requested=requested)
        assert resolve(cameras=cameras, prefill_w8a8=False, requested=256) == 256
    for requested in (0, -16, 273, True):
        with pytest.raises(ValueError, match="positive multiple of 16"):
            resolve(cameras=2, prefill_w8a8=True, requested=requested)
    with pytest.raises(ValueError, match="cameras must be 2 or 3"):
        resolve(cameras=1, prefill_w8a8=True, requested=None)


def test_cli_rejects_wrong_short_a8_chunk_before_loading_checkpoint(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_pi05_optimized.py", "--precision", "w4a16",
                                    "--cameras", "2", "--prefill-w8a8", "--prefill-chunk-size", "400",
                                    "--checkpoint", "/missing-checkpoint", "--batch", "/missing-batch.pt"])
    with pytest.raises(SystemExit) as error:
        RUNNER["main"]()
    assert error.value.code == 2
    assert "requires exactly --prefill-chunk-size 272" in capsys.readouterr().err


def _two_camera_policy_and_batch():
    class Config:
        empty_cameras = 1
        input_features = {"observation.images." + key: "visual" for key in ("image", "image2", "empty_camera_0")}
        input_features["observation.state"] = "state"

        @property
        def image_features(self):
            return {key: feature for key, feature in self.input_features.items() if feature == "visual"}

    lerobot = torch.nn.Module()
    lerobot.config = Config()
    lerobot.model = torch.nn.Module()
    lerobot.model.config = lerobot.config
    # Configuration must not dequantize or relabel the installed CPU checkpoint.
    lerobot.model.projection = torch.nn.Parameter(torch.arange(16, dtype=torch.uint8), requires_grad=False)
    lerobot._pi05_quant_config = {"method": "nvfp4a16", "nvfp4_abi": "striped_v2"}
    policy = SimpleNamespace(_lerobot_policy=lerobot, _adapter=SimpleNamespace(_lerobot_policy=lerobot))
    batch = {"observation.images." + key: torch.zeros(1, 3, 8, 8) for key in ("image", "image2")}
    batch["observation.language.tokens"] = torch.zeros(1, 32, dtype=torch.long)
    return policy, batch


def test_two_camera_configuration_is_shared_cold_and_preserves_quantized_weights():
    policy, batch = _two_camera_policy_and_batch()
    lerobot = policy._lerobot_policy
    config, weights, quant = lerobot.config, lerobot.model.projection, lerobot._pi05_quant_config
    receipt = RUNNER["configure_two_camera_policy"](policy, batch)
    assert config is lerobot.model.config
    assert list(config.image_features) == ["observation.images.image", "observation.images.image2"]
    assert config.empty_cameras == 0 and config.input_features["observation.state"] == "state"
    assert lerobot.model.projection is weights and weights.dtype == torch.uint8
    assert lerobot._pi05_quant_config is quant and quant["method"] == "nvfp4a16"
    assert receipt["before_empty_cameras"] == 1 and receipt["shared_config"] is True
    RUNNER["validate_camera_batch"](batch, config.image_features, cameras=2)


@pytest.mark.parametrize("bad_state", ["swizzle", "cache", "split_config"])
def test_two_camera_configuration_rejects_installed_or_split_policy_before_mutation(bad_state):
    policy, batch = _two_camera_policy_and_batch()
    lerobot = policy._lerobot_policy
    original = dict(lerobot.config.input_features)
    if bad_state == "swizzle":
        lerobot._rpu_swizzle_started = True
    elif bad_state == "cache":
        lerobot.model._rpu_fused_denoise_graph_cache = object()
    else:
        lerobot.model.config = object()
    with pytest.raises(ValueError, match="before|precede|cold"):
        RUNNER["configure_two_camera_policy"](policy, batch)
    assert lerobot.config.input_features == original and lerobot.config.empty_cameras == 1


def test_two_camera_configuration_rejects_empty_missing_extra_and_masked_slots():
    policy, batch = _two_camera_policy_and_batch()
    configure = RUNNER["configure_two_camera_policy"]
    first, second = "observation.images.image", "observation.images.image2"
    invalid = [
        {key: value for key, value in batch.items() if key != second},
        {**batch, "observation.images.empty_camera_0": batch[first]},
        {**batch, "observation.images.other": batch[first]},
        {**batch, second + "_padding_mask": torch.zeros(1, dtype=torch.bool)},
        {**batch, "observation.images.empty_camera_0_padding_mask": torch.ones(1, dtype=torch.bool)},
        {**batch, "observation.language.tokens": torch.zeros(1, 48, dtype=torch.long)},
    ]
    for invalid_batch in invalid:
        with pytest.raises(ValueError):
            configure(policy, invalid_batch)
        assert policy._lerobot_policy.config.empty_cameras == 1
        assert len(policy._lerobot_policy.config.image_features) == 3
    with pytest.raises(ValueError, match="real VISUAL"):
        RUNNER["validate_camera_batch"](
            {first: batch[first], "observation.images.empty_camera_0": batch[second],
             "observation.language.tokens": batch["observation.language.tokens"]},
            [first, "observation.images.empty_camera_0"], cameras=2)


def test_fp16_precomputed_adarms_table_does_not_call_weight_quantizer(monkeypatch):
    from rpu_backend.adapters.pi05.weights import _precompute_adarms_table
    from rpu_backend.quant import _common

    def reject_quantization(*args, **kwargs):
        raise AssertionError("FP16 conditioning must retain FP16 weights")
    monkeypatch.setattr(_common, "quantize_linear_per_channel", reject_quantization)
    dense = torch.nn.Linear(2, 6, dtype=torch.float16)
    with torch.no_grad():
        dense.weight.copy_(torch.tensor([[0.123, 1], [-0.25, 0.333], [0.75, -1],
                                         [0.111, 0.555], [-0.625, 1.5], [1.25, -0.125]]))
        dense.bias.zero_()
    expert = SimpleNamespace(layers=[], norm=SimpleNamespace(dense=dense))
    cond = torch.tensor([[0.5, -0.75], [-0.25, 0.125]], dtype=torch.float16)
    result = _precompute_adarms_table(expert, cond, dense_w8a16=False)
    expected = torch.nn.functional.linear(cond.float(), dense.weight.float()).half()
    expected[:, :2] += 1
    assert torch.equal(result[:, 0], expected)
    assert result.dtype == torch.float16


def test_fp16_mask_cache_binds_dtype_and_mask_content(monkeypatch):
    monkeypatch.setattr(runtime, "_prefix_mask_cache_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_PREFIX_MASK_SNAPSHOT", None)
    pads = torch.tensor([[True, True, False]])
    att = torch.zeros_like(pads)
    def prepare(value):
        return torch.where(value[:, None], 0.0, torch.finfo(torch.float32).min)
    mask32, _ = runtime._cached_build_prefix_mask_4d(pads, att, prepare)
    mask16, _ = runtime._cached_build_prefix_mask_4d(pads, att, prepare, output_dtype=torch.float16)
    assert mask16.dtype == torch.float16
    assert torch.equal(mask16, mask32.half())
    hit, _ = runtime._cached_build_prefix_mask_4d(pads, att, prepare, output_dtype=torch.float16)
    assert hit is mask16
    pads[0, 2] = True
    changed, _ = runtime._cached_build_prefix_mask_4d(pads, att, prepare, output_dtype=torch.float16)
    assert changed is not mask16 and not torch.equal(changed, mask16)


def test_public_prepare_forwards_table_option_and_rejects_nonbool():
    from rpu_backend.api.policy import Pi05Policy
    policy = object.__new__(Pi05Policy)
    policy._rpu_execution = {}
    policy._rpu_ready = True
    policy._lerobot_policy = torch.nn.Module()
    calls = []
    policy._adapter = SimpleNamespace(prepare_graphs=lambda batch, **kwargs: calls.append(kwargs))
    policy.prepare_graphs({}, num_steps=10, precompute_adarms=True)
    assert calls == [{"num_steps": 10, "precompute_adarms": True}]
    with pytest.raises(TypeError, match="must be bool"):
        policy.prepare_graphs({}, precompute_adarms="false")
    assert len(calls) == 1


@pytest.mark.parametrize("text_tokens,pair_rows", [(64, 416), (96, 432), (128, 448), (160, 464)])
def test_text_bucket_is_explicit_cold_and_ignores_checkpoint_tokenizer_default(text_tokens, pair_rows):
    from rpu_backend.adapters.pi05 import _selected_prefill_pair_rows
    policy, _ = _two_camera_policy_and_batch()
    policy._lerobot_policy.config.tokenizer_max_length = 200
    projection = policy._lerobot_policy.model.projection
    assert _selected_prefill_pair_rows(policy._adapter) == 400
    receipt = RUNNER["configure_text_profile"](policy, text_tokens)
    assert receipt["prefill_pair_rows"] == _selected_prefill_pair_rows(policy._adapter) == pair_rows
    assert policy._lerobot_policy.config.tokenizer_max_length == 200
    assert policy._lerobot_policy.model.projection is projection
    policy._adapter._rpu_is_ready = True
    with pytest.raises(ValueError, match="precede device installation"):
        RUNNER["configure_text_profile"](policy, 32)
    assert policy._adapter._rpu_prefill_text_tokens == text_tokens


@pytest.mark.parametrize("invalid", [True, 48, 200, 256, None])
def test_text_bucket_rejects_invalid_selection_without_mutation(invalid):
    policy, _ = _two_camera_policy_and_batch()
    with pytest.raises(ValueError, match="text_tokens must be"):
        RUNNER["configure_text_profile"](policy, invalid)
    assert not hasattr(policy._adapter, "_rpu_prefill_text_tokens")


def test_t64_selection_rejects_unadmitted_cameras_precision_and_cores():
    from rpu_backend.adapters.pi05 import _selected_prefill_pair_rows
    policy, batch = _two_camera_policy_and_batch()
    RUNNER["configure_two_camera_policy"](policy, batch)
    assert _selected_prefill_pair_rows(policy._adapter) == 272
    RUNNER["configure_text_profile"](policy, 64)
    assert _selected_prefill_pair_rows(policy._adapter) == 288
    del policy._lerobot_policy.config.input_features["observation.images.image2"]
    with pytest.raises(ValueError, match="two or three cameras"):
        _selected_prefill_pair_rows(policy._adapter)
    for precision in ("int4_g128", "legacy_int4"):
        with pytest.raises(ValueError, match="FP16, W8A16, or Action NVFP4"):
            RUNNER["profile_environment"](precision, text_tokens=64)
    policy, _ = _two_camera_policy_and_batch()
    policy._adapter._rpu_prefill_text_tokens = 64
    policy._adapter._rpu_execution = {"model": {"num_cores": 6}}
    with pytest.raises(ValueError, match="eight cores"):
        _selected_prefill_pair_rows(policy._adapter)


@pytest.mark.parametrize("text_tokens,pair_rows", [(64, 416), (96, 432), (128, 448), (160, 464)])
def test_extended_text_requires_exact_pair_and_right_padded_mask_without_changing_t32_flags(text_tokens, pair_rows):
    for precision in ("fp16", "w8a16", "w4a16"):
        kwargs = dict(vision_owner_ln=True, prefill_w8a8=precision == "w4a16")
        assert RUNNER["profile_environment"](precision, text_tokens=text_tokens, **kwargs) == RUNNER["profile_environment"](precision, **kwargs)
        assert RUNNER["resolve_prefill_chunk_size"](cameras=3, prefill_w8a8=False, requested=None, text_tokens=text_tokens) == pair_rows
        with pytest.raises(ValueError, match=f"requires exactly --prefill-chunk-size {pair_rows}"):
            RUNNER["resolve_prefill_chunk_size"](cameras=3, prefill_w8a8=False, requested=400, text_tokens=text_tokens)
    cameras = {"observation.images." + name: object() for name in ("base", "left", "right")}
    batch = {key: torch.zeros(1, 3, 8, 8) for key in cameras}
    batch["observation.language.tokens"] = torch.arange(text_tokens).reshape(1, text_tokens)
    mask = torch.arange(text_tokens).reshape(1, text_tokens) < text_tokens - 16
    batch["observation.language.attention_mask"] = mask
    RUNNER["validate_camera_batch"](batch, cameras, cameras=3, text_tokens=text_tokens)
    for bad in (torch.ones(1, text_tokens), torch.zeros(1, text_tokens, dtype=torch.bool), torch.roll(mask, 1, 1)):
        with pytest.raises(ValueError, match="right padding"):
            RUNNER["validate_camera_batch"]({**batch, "observation.language.attention_mask": bad}, cameras, cameras=3, text_tokens=text_tokens)
    with pytest.raises(ValueError, match="32 language tokens"):
        RUNNER["validate_three_camera_batch"](batch, cameras)


def test_cli_rejects_wrong_two_camera_t64_chunk_before_loading_checkpoint(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_pi05_optimized.py", "--precision", "fp16",
                                    "--cameras", "2", "--text-tokens", "64", "--prefill-chunk-size", "272",
                                    "--checkpoint", "/missing-checkpoint", "--batch", "/missing-batch.pt"])
    with pytest.raises(SystemExit) as error:
        RUNNER["main"]()
    assert error.value.code == 2
    assert "2-camera T64 requires exactly --prefill-chunk-size 288" in capsys.readouterr().err


@pytest.mark.parametrize("text_tokens,valid_tokens,pair_rows", [(64, 48, 288), (96, 65, 304), (128, 96, 320), (160, 159, 336)])
def test_two_camera_long_text_configures_images_before_text_without_mutating_checkpoint(text_tokens, valid_tokens, pair_rows):
    from rpu_backend.adapters.pi05 import _selected_prefill_pair_rows
    policy, batch = _two_camera_policy_and_batch()
    batch["observation.language.tokens"] = torch.arange(text_tokens).reshape(1, text_tokens)
    batch["observation.language.attention_mask"] = torch.arange(text_tokens).reshape(1, text_tokens) < valid_tokens
    lerobot = policy._lerobot_policy
    weight, quant = lerobot.model.projection, lerobot._pi05_quant_config
    before = dict(lerobot.config.input_features)
    with pytest.raises(ValueError, match="32 language tokens"):
        RUNNER["configure_two_camera_policy"](policy, batch)
    assert lerobot.config.input_features == before
    camera_receipt = RUNNER["configure_two_camera_policy"](policy, batch, text_tokens=text_tokens)
    text_receipt = RUNNER["configure_text_profile"](policy, text_tokens)
    assert len(camera_receipt["image_features"]) == 2 and lerobot.config.empty_cameras == 0
    assert text_receipt["prefill_pair_rows"] == _selected_prefill_pair_rows(policy._adapter) == pair_rows
    assert lerobot.model.projection is weight and lerobot._pi05_quant_config is quant
    assert RUNNER["resolve_prefill_chunk_size"](cameras=2, prefill_w8a8=True, requested=None, text_tokens=text_tokens) == pair_rows


@pytest.mark.parametrize("text_tokens", [64, 96, 128, 160])
@pytest.mark.parametrize("bad_state", ["mask", "hot"])
def test_two_camera_long_text_rejects_bad_input_before_camera_or_text_mutation(bad_state, text_tokens):
    policy, batch = _two_camera_policy_and_batch()
    batch["observation.language.tokens"] = torch.arange(text_tokens).reshape(1, text_tokens)
    batch["observation.language.attention_mask"] = torch.ones(1, text_tokens, dtype=torch.bool)
    before = dict(policy._lerobot_policy.config.input_features)
    if bad_state == "mask":
        batch["observation.language.attention_mask"][0, 8] = False
    else:
        policy._adapter._rpu_is_ready = True
    with pytest.raises(ValueError):
        RUNNER["configure_two_camera_policy"](policy, batch, text_tokens=text_tokens)
    assert policy._lerobot_policy.config.input_features == before
    assert policy._lerobot_policy.config.empty_cameras == 1
    assert not hasattr(policy._adapter, "_rpu_prefill_text_tokens")


@pytest.mark.parametrize("text_tokens", [96, 128, 160])
def test_extended_text_one_camera_selection_fails_before_partial_mutation(text_tokens):
    from rpu_backend.adapters.pi05 import _selected_prefill_pair_rows
    policy, _ = _two_camera_policy_and_batch()
    features = policy._lerobot_policy.config.input_features
    del features["observation.images.empty_camera_0"]
    del features["observation.images.image2"]
    before = dict(features)
    with pytest.raises(ValueError, match="two or three cameras"):
        RUNNER["configure_text_profile"](policy, text_tokens)
    assert features == before
    assert not hasattr(policy._adapter, "_rpu_prefill_text_tokens")
    policy._adapter._rpu_prefill_text_tokens = text_tokens
    with pytest.raises(ValueError, match="two or three cameras"):
        _selected_prefill_pair_rows(policy._adapter)


@pytest.mark.parametrize("text_tokens", [96, 128, 160])
def test_extended_text_rejects_legacy_precision_and_non_eight_core_install(text_tokens):
    from rpu_backend.adapters.pi05 import _selected_prefill_pair_rows
    for precision in ("int4_g128", "legacy_int4"):
        with pytest.raises(ValueError, match="FP16, W8A16, or Action NVFP4"):
            RUNNER["profile_environment"](precision, text_tokens=text_tokens)
    policy, _ = _two_camera_policy_and_batch()
    policy._lerobot_policy._pi05_quant_config = {"method": "w4a16"}
    with pytest.raises(ValueError, match="FP16, W8A16, or Action NVFP4"):
        RUNNER["configure_text_profile"](policy, text_tokens)
    assert not hasattr(policy._adapter, "_rpu_prefill_text_tokens")
    policy._adapter._rpu_prefill_text_tokens = text_tokens
    for cores in (4, 6):
        policy._adapter._rpu_execution = {"model": {"num_cores": cores}}
        with pytest.raises(ValueError, match="eight cores"):
            _selected_prefill_pair_rows(policy._adapter)


@pytest.mark.parametrize("text_tokens,cameras,chunk,message", [(160, 3, 448, "requires exactly --prefill-chunk-size 464"),
                                                              (160, 2, 320, "requires exactly --prefill-chunk-size 336"),
                                                              (128, 3, 416, "requires exactly --prefill-chunk-size 448"),
                                                              (128, 2, 288, "requires exactly --prefill-chunk-size 320"),
                                                              (96, 3, 416, "requires exactly --prefill-chunk-size 432"),
                                                              (96, 2, 288, "requires exactly --prefill-chunk-size 304")])
def test_extended_text_cli_rejects_unadmitted_geometry_before_checkpoint_loading(monkeypatch, capsys, text_tokens, cameras, chunk, message):
    monkeypatch.setattr("sys.argv", ["run_pi05_optimized.py", "--precision", "fp16",
                                    "--cameras", str(cameras), "--text-tokens", str(text_tokens),
                                    "--prefill-chunk-size", str(chunk),
                                    "--checkpoint", "/missing-checkpoint", "--batch", "/missing-batch.pt"])
    with pytest.raises(SystemExit) as error:
        RUNNER["main"]()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_per_channel_packing_honors_explicit_group_size(group_size):
    from rpu_backend.quant.int4_pgrp_pack import pack_int4_per_channel_as_pgrp

    module = torch.nn.Linear(1024, 256, bias=False)
    codes = torch.arange(256 * 1024).reshape(256, 1024).remainder(16).to(torch.int8) - 8
    scale = torch.ones(256, dtype=torch.float16)
    module.weight = torch.nn.Parameter(codes, requires_grad=False)
    module.register_buffer("weight_scale", scale)
    decoder = SimpleNamespace(layers=[SimpleNamespace(self_attn=torch.nn.Module())])
    decoder.layers[0].self_attn.q_proj = module
    expected, expected_scale = pack_int4_per_channel_as_pgrp(codes, scale, 1, 8,
                                                            group_size=group_size)
    assert pack_int4_projections_inplace(decoder, {"q_proj"}, attn_num_cores=8,
                                         group_size=group_size) == 1
    assert torch.equal(module.weight, expected.reshape(256, 512))
    assert torch.equal(module.weight_scale, expected_scale)
    assert module.weight_scale.shape[0] == group_size
    assert pack_int4_projections_inplace(decoder, {"q_proj"}, attn_num_cores=8,
                                         group_size=group_size) == 0


def test_grouped_packing_rejects_a_different_requested_group_size():
    module = torch.nn.Linear(1024, 256, bias=False)
    module.weight = torch.nn.Parameter(torch.zeros(256, 1024, dtype=torch.int8), requires_grad=False)
    module.register_buffer("weight_scale", torch.ones(8, 256, dtype=torch.float16))
    attn = torch.nn.Module()
    attn.q_proj = module
    decoder = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
    with pytest.raises(RuntimeError, match="logical group scales/metadata mismatch"):
        pack_int4_projections_inplace(decoder, {"q_proj"}, attn_num_cores=8, group_size=32)
    assert module.weight.dtype == torch.int8
