"""Exercise real MoE installation and forward boundaries without a board."""
from contextlib import nullcontext
from types import SimpleNamespace
import json

import pytest
import torch


@pytest.mark.parametrize("logical,physical,chunk", [(63, 64, 64), (1, 1, 0)])
def test_real_forward_uses_descriptor_authority_and_preserves_receipt(
    monkeypatch, logical, physical, chunk,
):
    from rpu_backend.adapters.qwen3_5_moe import text
    import rpu_backend

    monkeypatch.setattr(rpu_backend.graph, "GraphSignature",
                        lambda **kwargs: SimpleNamespace(**kwargs))

    calls = []
    signatures = []
    descriptor = (123, 456)
    graph_words = (17, 29)
    plan = SimpleNamespace(
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(
            physical_descriptor=descriptor)),
        graph_mode="BOUNDED_ONESHOT",
        graph_key_words=lambda: graph_words,
        as_dict=lambda **_: {"plan_digest": "p", "domain_digest": "d",
                            "optimality": "EXACT_DOMAIN", "search_complete": True},
    )
    graph = SimpleNamespace(begin=lambda: None, end=lambda: None, abort=lambda: None)
    state = SimpleNamespace(
        handle=17, mrope_section=(), prefill_graph=graph,
        graph_cache=SimpleNamespace(
            capture=lambda sig: signatures.append(sig) or nullcontext()),
        hidden_size=4, num_layers=40, last_a6_plan=None,
    )
    model = SimpleNamespace(_rpu_qwen3_5_moe=state)
    cache = SimpleNamespace(position=0, k_caches=[], v_caches=[],
                            gdn_states=[], conv_states=[])
    cache.update_position = lambda n: setattr(cache, "position", cache.position + n)

    def forward(*args):
        calls.append(args)
        assert args[-2] == 0  # Explicit chunk plus descriptor is rejected natively.
        assert args[-1] == descriptor
        assert args[1].shape == (1, physical, 4)
        if physical > logical:
            assert not torch.count_nonzero(args[1][:, logical:])
        return args[1] + 1

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_forward", forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_set_valid_prefill_len",
                        lambda handle, n: None, raising=False)
    monkeypatch.setattr(text, "_plan_qwen3_5_moe_decode", lambda _: plan)
    original_to = torch.Tensor.to
    uploads = []

    def upload(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            uploads.append(tensor.clone())
            return tensor.clone()  # Native device/publication boundary double.
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", upload)
    output = text.run_qwen3_5_moe_text(
        model, torch.ones(1, logical, 4, dtype=torch.float16), cache,
        preflight_plan=(physical, chunk, plan),
        _private_cpu_prefill=logical > 1,
    )
    assert len(calls) == 1 and cache.position == logical
    assert len(uploads) == (1 if logical > 1 else 0)
    if uploads:
        assert uploads[0].shape == (1, physical, 4)
        assert torch.all(uploads[0][:, :logical] == 1)
        assert not uploads[0][:, logical:].count_nonzero()
    assert output.shape == (1, logical, 4) and torch.all(output == 2)
    receipt = model._rpu_last_execution_plan
    assert receipt["physical_descriptor"] == descriptor
    assert receipt["chunk_size"] == chunk
    assert receipt["execution_len"] == physical
    if logical == 1:
        assert len(signatures) == 1
        assert signatures[0].dyn_dims == [40, 0, *graph_words]
    else:
        assert signatures == []


def test_real_prefill_rejects_missing_descriptor_before_dispatch(monkeypatch):
    from rpu_backend.adapters.qwen3_5_moe import text

    model = SimpleNamespace(_rpu_qwen3_5_moe=SimpleNamespace(handle=17))
    cache = SimpleNamespace(position=0)
    plan = SimpleNamespace(selected=SimpleNamespace(
        stage_tuple=SimpleNamespace(physical_descriptor=())))
    called = []
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_forward",
                        lambda *args: called.append(args), raising=False)
    with pytest.raises(RuntimeError, match="no native stage descriptor"):
        text.run_qwen3_5_moe_text(
            model, torch.ones(1, 63, 4, dtype=torch.float16), cache,
            preflight_plan=(64, 64, plan),
        )
    assert not called and cache.position == 0


def test_moe_installed_state_passes_attribute_guard_but_user_injection_does_not():
    from rpu_backend.api.errors import RPUConfigError
    from rpu_backend.runtime.hw_attrs import (
        install_hw_attr_validator, validate_postinstall, validate_preinstall,
    )

    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    inner = model.language_model
    inner._rpu_qwen3_5_moe_text_install_started = True
    inner._rpu_qwen3_5_moe = SimpleNamespace(handle=17)
    with pytest.raises(RPUConfigError):
        validate_preinstall(model)
    validate_postinstall(model)
    install_hw_attr_validator(model)
    with pytest.raises(RPUConfigError, match="_rpu_qwen3_5_moe_typo"):
        inner._rpu_qwen3_5_moe_typo = True


@pytest.mark.parametrize("subfolder", ("", "mixed"))
def test_loader_preserves_checkpoint_generation_config(tmp_path, monkeypatch, subfolder):
    from safetensors.torch import save_file
    from transformers import GenerationConfig
    from rpu_backend.adapters.qwen3_5_moe import loader

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)
            self.generation_config = GenerationConfig(eos_token_id=248044)

        def tie_weights(self):
            pass

    # Geometry/quantization admission is covered separately; retain the real
    # loader, real safetensors assignment and real generation sidecar handling.
    monkeypatch.setattr(loader, "parse_quant_config", lambda _: {"profile": "fixture"})
    monkeypatch.setattr(loader, "_layer_types", lambda _: ())
    monkeypatch.setattr(loader, "expected_fp8_weight_names", lambda _: ())
    monkeypatch.setattr(loader.AutoModelForImageTextToText, "from_config", lambda _, **kwargs: TinyModel())
    checkpoint = tmp_path / subfolder
    checkpoint.mkdir(exist_ok=True)
    save_file({"linear.weight": torch.ones(2, 2)}, checkpoint / "model.safetensors")
    (checkpoint / "generation_config.json").write_text(json.dumps({
        "eos_token_id": [248046, 248044], "pad_token_id": 248044,
    }))
    if subfolder:
        # A valid parent must not silently supply another checkpoint's weights
        # or stopping tokens when the selected payload is nested.
        save_file({"linear.weight": torch.zeros(2, 2)}, tmp_path / "model.safetensors")
        (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    model = loader.load_qwen3_5_moe_model(tmp_path, config=SimpleNamespace(), subfolder=subfolder)
    assert model.generation_config.eos_token_id == [248046, 248044]
    assert model.generation_config.pad_token_id == 248044
    assert torch.equal(model.linear.weight, torch.ones(2, 2, dtype=torch.float16))


@pytest.mark.parametrize("capacity", [512, 8193, 16384, 32768])
def test_valid_install_resolves_options_before_hardware(monkeypatch, capacity):
    from rpu_backend.adapters import qwen3_5_moe as moe
    from rpu_backend.runtime import hw_attrs

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model._rpu_execution = {"model": {"num_cores": 8},
                            "prefill": {"chunk_size": "auto"},
                            "vision": {"chunk_size": "auto"}}
    adapter = object.__new__(moe.Qwen3_5MoeAdapter)
    adapter.model = model
    adapter._rpu_is_ready = False

    def before_hardware(actual):
        assert actual is model
        raise RuntimeError("reached preinstall boundary")

    monkeypatch.setattr(hw_attrs, "validate_preinstall", before_hardware)
    with pytest.raises(RuntimeError, match="reached preinstall boundary"):
        adapter.to_rpu(max_seq_len=capacity)
    assert not getattr(model, "_rpu_swizzle_started", False)
    assert not moe._SWIZZLE_LOCK.locked()


@pytest.mark.parametrize("cores", [4, 8])
def test_real_moe_weight_import_and_expert_split(cores):
    # Import inside the test: metadata-only adapter imports never reached this
    # install-time dependency and previously missed a missing quant.fp8 module.
    from rpu_backend.adapters.qwen3_5_moe.weights import _prepare_routed

    experts, intermediate, hidden = 2, 256, 256
    gate_up = torch.empty(experts, 2 * intermediate, hidden, dtype=torch.uint8)
    gate_up[:, :intermediate].fill_(0x38)  # E4M3 1.0
    gate_up[:, intermediate:].fill_(0x40)  # E4M3 2.0
    original = gate_up.clone()
    scales = torch.ones(experts, 2 * intermediate, dtype=torch.float16)
    scales[:, intermediate:] = 3
    module = SimpleNamespace(
        gate_up_proj=gate_up,
        down_proj=torch.full((experts, hidden, intermediate), 0x30, dtype=torch.uint8),
        gate_up_proj_scale=scales,
        down_proj_scale=torch.full((experts, hidden), 4, dtype=torch.float16),
        rpu_fp8_format="fp8_e4m3", rpu_weight_mode=5, rpu_fp8_scale_axis=-1,
    )
    gate, up, down, gate_scale, up_scale, down_scale, mode = _prepare_routed(module, cores)
    assert gate.shape == up.shape == down.shape == (experts, intermediate * hidden)
    assert torch.all(gate == 0x38) and torch.all(up == 0x40) and torch.all(down == 0x30)
    assert torch.all(gate_scale == 1) and torch.all(up_scale == 3)
    assert torch.all(down_scale == 4) and mode == 5
    assert torch.equal(module.gate_up_proj, original)


def test_fp8_zero_rows_and_signed_values_remain_finite():
    from rpu_backend.quant.fp8 import (
        dequantize_fp8_e4m3_per_output, quantize_fp8_e4m3_per_output,
    )

    source = torch.tensor([[0, 0, 0, 0], [-448, -224, 224, 448]], dtype=torch.float32)
    codes, scales = quantize_fp8_e4m3_per_output(source)
    actual = dequantize_fp8_e4m3_per_output(codes, scales)
    assert codes.dtype == torch.uint8 and scales.dtype == torch.float16
    assert torch.all(torch.isfinite(scales)) and torch.all(scales > 0)
    assert torch.equal(actual, source)


def test_tp6_cold_padding_preserves_split_expert_math_and_source():
    from rpu_backend.adapters.qwen3_5_moe.weights import _prepare_routed
    from rpu_backend.quant.fp8 import dequantize_fp8_e4m3_per_output as decode

    generator = torch.Generator().manual_seed(176)
    e, h, logical, physical, cores = 2, 64, 512, 576, 6
    gate_up = torch.randint(0, 96, (e, logical * 2, h), dtype=torch.uint8, generator=generator)
    down = torch.randint(0, 96, (e, h, logical), dtype=torch.uint8, generator=generator)
    scales = torch.rand(e, logical * 2, generator=generator).half() + 0.25
    down_scales = torch.rand(e, h, generator=generator).half() + 0.25
    module = SimpleNamespace(gate_up_proj=gate_up, down_proj=down,
        gate_up_proj_scale=scales, down_proj_scale=down_scales,
        rpu_fp8_format="fp8_e4m3", rpu_weight_mode=5, rpu_fp8_scale_axis=-1)
    source = [t.clone() for t in (gate_up, down, scales, down_scales)]
    gate, up, packed_down, gs, us, ds, mode = _prepare_routed(module, cores)
    # Decode the public grouped ABI independently of the padding helper, then
    # compare represented mathematical projections, including new zero lanes.
    def unpack_col(tensor):
        return tensor.view(e, physical // cores // 16, h // 32, cores, 16, 32).permute(
            0, 3, 1, 4, 2, 5).contiguous().view(e, physical, h)
    raw_gate, raw_up = unpack_col(gate), unpack_col(up)
    raw_down = packed_down.view(e, h // 16, physical // cores // 32, cores, 16, 32).permute(
        0, 1, 4, 3, 2, 5).contiguous().view(e, h, physical)
    assert mode == 5
    for padded, original in ((raw_gate, source[0][:, :logical]),
                             (raw_up, source[0][:, logical:])):
        assert torch.equal(padded[:, :logical], original)
        assert torch.count_nonzero(padded[:, logical:]) == 0
    assert torch.equal(raw_down[:, :, :logical], source[1])
    assert torch.count_nonzero(raw_down[:, :, logical:]) == 0
    assert torch.equal(gs[:, :logical], source[2][:, :logical])
    assert torch.equal(us[:, :logical], source[2][:, logical:])
    assert torch.all(gs[:, logical:] == 1) and torch.all(us[:, logical:] == 1)
    assert torch.equal(ds, source[3])
    x = torch.randn(e, h, 1, generator=generator)
    expected_gate = decode(source[0][:, :logical], source[2][:, :logical]) @ x
    expected_up = decode(source[0][:, logical:], source[2][:, logical:]) @ x
    actual_gate, actual_up = decode(raw_gate, gs) @ x, decode(raw_up, us) @ x
    torch.testing.assert_close(actual_gate[:, :logical], expected_gate, rtol=0, atol=0)
    torch.testing.assert_close(actual_up[:, :logical], expected_up, rtol=0, atol=0)
    actual = decode(raw_down, ds).double() @ (actual_gate.double() * actual_up.double())
    expected = decode(source[1], source[3]).double() @ (expected_gate.double() * expected_up.double())
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-7)
    for actual_source, original in zip((gate_up, down, scales, down_scales), source):
        assert torch.equal(actual_source, original)


def test_tp6_full_cold_preparation_keeps_router4_and_shared6_math():
    from rpu_backend.adapters.qwen3_5_moe.weights import prepare_moe_layer_weights

    rng = torch.Generator().manual_seed(621)
    h, e, logical, physical = 64, 2, 512, 576
    def fp16(*shape):
        return (torch.randn(*shape, generator=rng) * 0.05).half()
    expert = SimpleNamespace(
        gate_up_proj=torch.zeros(e, logical * 2, h, dtype=torch.uint8),
        down_proj=torch.zeros(e, h, logical, dtype=torch.uint8),
        gate_up_proj_scale=torch.ones(e, logical * 2, dtype=torch.float16),
        down_proj_scale=torch.ones(e, h, dtype=torch.float16),
        rpu_fp8_format="fp8_e4m3", rpu_weight_mode=5, rpu_fp8_scale_axis=-1)
    raw = {"router": fp16(256, h), "gate": fp16(logical, h),
           "up": fp16(logical, h), "down": fp16(h, logical),
           "scalar": fp16(1, h)}
    original = {name: tensor.clone() for name, tensor in raw.items()}
    mlp = SimpleNamespace(experts=expert, gate=SimpleNamespace(weight=raw["router"]),
        shared_expert=SimpleNamespace(
            gate_proj=SimpleNamespace(weight=raw["gate"]),
            up_proj=SimpleNamespace(weight=raw["up"]),
            down_proj=SimpleNamespace(weight=raw["down"])),
        shared_expert_gate=SimpleNamespace(weight=raw["scalar"]))
    packed = prepare_moe_layer_weights(mlp, num_cores=6, device="cpu")

    def col(tensor, cores):
        n, k = tensor.shape
        return tensor.view(n // 16 // cores, k // 16, cores, 16, 16).permute(
            2, 0, 3, 1, 4).contiguous().view(n, k)
    router = col(packed.router, 4)
    gate, up = col(packed.shared_gate, 6), col(packed.shared_up, 6)
    down = packed.shared_down.view(h // 16, physical // 16 // 6, 6, 16, 16).permute(
        0, 3, 2, 1, 4).contiguous().view(h, physical)
    scalar = col(packed.shared_scalar_gate, 6)
    assert torch.equal(router, original["router"])
    assert torch.equal(scalar, original["scalar"].expand(16 * 6, -1))
    for expanded, source in ((gate, original["gate"]), (up, original["up"])):
        assert expanded.shape == (physical, h)
        assert torch.equal(expanded[:logical], source)
        assert torch.count_nonzero(expanded[logical:]) == 0
    assert torch.equal(down[:, :logical], original["down"])
    assert torch.count_nonzero(down[:, logical:]) == 0
    x = torch.randn(h, 3, generator=rng, dtype=torch.float64)
    def project(g, u, d):
        return d.double() @ (torch.nn.functional.silu(g.double() @ x) * (u.double() @ x))
    torch.testing.assert_close(project(gate, up, down),
        project(original["gate"], original["up"], original["down"]), rtol=1e-12, atol=1e-12)
    assert all(torch.equal(raw[name], tensor) for name, tensor in original.items())


@pytest.mark.parametrize("partition,shape", [
    ("col", (2, 255, 256)), ("col", (2, 256, 255)),
    ("row", (2, 255, 256)), ("row", (2, 256, 255)),
])
def test_grouped_packing_rejects_incomplete_hardware_tiles(partition, shape):
    from rpu_backend.quant.fp8 import pack_grouped_weight

    with pytest.raises(ValueError, match="violates"):
        pack_grouped_weight(torch.zeros(shape, dtype=torch.uint8),
                            partition=partition, num_cores=8)
