"""Frozen CPU codec geometry, overlap boundaries, and output ownership."""
import pytest
import torch
import torch.nn.functional as F

from rpu_backend.adapters.qwen3_tts.codec import (
    install_codec_activations, install_codec_convolutions,
    install_rpu_convolutions, install_rpu_residual_convolutions,
    _residual_convolution, _rpu_convolution, _rpu_linear,
    _rpu_transpose_convolution,
)


def test_spm_convolution_passes_prepared_bias_without_host_add(monkeypatch):
    module = torch.nn.Conv1d(96, 1, 7)
    hidden, weight = torch.randn(1, 96, 9), torch.randn(128, 672).half()
    bias, output = torch.randn(128).half(), torch.randn(1, 1, 3)
    def native(x, w, b, kernel, dilation, channels):
        assert x is hidden and w is weight and b is bias
        assert (kernel, dilation, channels) == (7, 1, 1)
        return output
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv1d_spm_bias", native, raising=False)
    stats = {"calls": 0, "spm_im2col_calls": 0}
    assert _rpu_convolution(module, hidden, weight, stats, bias) is output
    assert stats == {"calls": 1, "spm_im2col_calls": 1}


class SnakeBeta(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.randn(5))
        self.beta = torch.nn.Parameter(torch.randn(5))
        self.no_div_by_zero = 1e-9

    def forward(self, hidden):
        alpha = self.alpha[None, :, None].exp()
        beta = self.beta[None, :, None].exp()
        return hidden + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(hidden * alpha), 2)


@pytest.mark.parametrize("length", [1, 7, 1024])
def test_snake_matches_frozen_formula_without_mutating_input_or_retained_output(length):
    torch.manual_seed(12)
    module = SnakeBeta().eval()
    hidden = torch.randn(1, 5, length * 2)[:, :, ::2]
    saved = hidden.clone()
    original, restore = module.forward, []
    with torch.inference_mode():
        expected = module(hidden)
        install_codec_activations(module, restore)
        result = module(hidden)
        assert torch.equal(result, expected) and torch.equal(hidden, saved)
        module(hidden + 1)
        assert torch.equal(result, expected)
        assert restore == [(module, original)]
        module.forward = restore[0][1]
        assert torch.equal(module(hidden), expected)


def test_snake_rejects_parameters_before_patching_any_module():
    decoder = torch.nn.Sequential(SnakeBeta(), SnakeBeta().half())
    originals, restore = [m.forward for m in decoder], []
    with pytest.raises(ValueError, match="CPU FP32"):
        install_codec_activations(decoder, restore)
    assert not restore and [m.forward for m in decoder] == originals


@pytest.mark.parametrize("kernel,stride", [(2, 2), (16, 8), (10, 5), (8, 4), (6, 3)])
@pytest.mark.parametrize("length", [1, 7])
@pytest.mark.parametrize("bias", [False, True])
def test_transpose_matches_previous_fold_at_boundaries(kernel, stride, length, bias):
    torch.manual_seed(12)
    module = torch.nn.ConvTranspose1d(3, 5, kernel, stride=stride, bias=bias).eval()
    hidden = torch.randn(1, 3, length * 2)[:, :, ::2]
    saved = hidden.clone()
    restore = []
    with torch.inference_mode():
        projected = hidden.transpose(1, 2) @ module.weight.flatten(1)
        expected = F.fold(projected.transpose(1, 2),
            (1, (length - 1) * stride + kernel), (1, kernel), stride=(1, stride)).squeeze(2)
        if module.bias is not None:
            expected += module.bias[None, :, None]
        install_codec_convolutions(module, restore)
        result = module(hidden)
        assert result.shape == expected.shape and result.dtype == torch.float32
        assert torch.isfinite(result).all() and torch.equal(result, expected)
        assert torch.equal(hidden, saved)
        held = result.clone()
        module(hidden + 1)
        assert torch.equal(result, held)
        with pytest.raises(ValueError, match="output size"):
            module(hidden, output_size=expected.shape)


@pytest.mark.parametrize("length", [1, 7, 70])
@pytest.mark.parametrize("bias", [False, True])
def test_pointwise_matches_original_and_restores(length, bias):
    torch.manual_seed(12)
    module = torch.nn.Conv1d(3, 5, 1, bias=bias).eval()
    hidden = torch.randn(1, 3, length)
    saved = hidden.clone()
    original = module.forward
    restore = []
    with torch.inference_mode():
        expected = module(hidden)
        previous = module.weight.squeeze(-1) @ hidden
        if module.bias is not None:
            previous = previous + module.bias[None, :, None]
        install_codec_convolutions(module, restore)
        result = module(hidden)
        assert result.shape == expected.shape and result.dtype == torch.float32
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, expected)
        assert torch.equal(result, previous) and torch.equal(hidden, saved)
        held = result.clone()
        module(hidden + 1)
        assert torch.equal(result, held)
        assert restore == [(module, original)]
        module.forward = restore[0][1]
        assert torch.equal(module(hidden), expected)


@pytest.mark.parametrize("invalid", [
    torch.nn.ConvTranspose1d(4, 4, 3, stride=2),
    torch.nn.Conv1d(4, 4, 1, stride=2),
    torch.nn.Conv1d(4, 4, 1, groups=2),
])
def test_geometry_rejected_before_any_patch(invalid):
    modules = torch.nn.Sequential(torch.nn.ConvTranspose1d(4, 4, 2, stride=2), invalid)
    originals = [m.forward for m in modules]
    restore = []
    with pytest.raises(ValueError, match="geometry"):
        install_codec_convolutions(modules, restore)
    assert restore == [] and [m.forward for m in modules] == originals


@pytest.mark.parametrize("channels,out_channels,kernel,dilation,groups", [
    (96, 96, 7, 1, 1), (96, 96, 7, 3, 1), (96, 96, 7, 9, 1),
    (16, 32, 1, 1, 1), (32, 16, 3, 1, 1), (96, 1, 7, 1, 1),
    (16, 16, 7, 1, 16),
])
def test_rpu_im2col_matches_rounded_conv_and_owns_outputs(
        monkeypatch, channels, out_channels, kernel, dilation, groups):
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
        self if a == ("rpu",) else original_to(self, *a, **kw))
    monkeypatch.delattr(torch.ops.rpu, "tts_codec_conv1d_spm", raising=False)
    monkeypatch.delattr(torch.ops.rpu, "tts_codec_conv1d_spm_bias", raising=False)
    monkeypatch.delattr(torch.ops.rpu, "tts_codec_conv1d", raising=False)
    seen = []
    def linear(hidden, weight, bias, partition, cores, acc32):
        assert hidden.dtype == weight.dtype == torch.float16
        assert (bias, partition, cores, acc32) == (None, 1, 8, True)
        seen.append(hidden.clone())
        return (hidden.float() @ weight.float().T).half()
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", linear, raising=False)
    torch.manual_seed(1)
    module = torch.nn.Conv1d(channels, out_channels, kernel, dilation=dilation, groups=groups).eval()
    hidden = torch.randn(1, channels, 2 * ((kernel - 1) * dilation + 11))[:, :, ::2]
    source = hidden.clone()
    weight = module.weight.detach().flatten(1).half()
    if groups != 1:
        weight = torch.block_diag(*(row.unsqueeze(0) for row in weight))
    weight = F.pad(weight, (0, 0, 0, (-out_channels) % 128))
    stats = {"calls": 0}
    with torch.inference_mode():
        result = _rpu_convolution(module, hidden, weight, stats)
        patches = hidden.unfold(-1, (kernel - 1) * dilation + 1, 1)[..., ::dilation]
        assert torch.equal(seen[0], patches.permute(0, 2, 1, 3).reshape(11, -1).half())
        expected = F.conv1d(hidden.half().float(), module.weight.half().float(), dilation=dilation, groups=groups).half().float()
        expected += module.bias[None, :, None]
        torch.testing.assert_close(result, expected, rtol=0.002, atol=0.001)
        assert result.shape == (1, out_channels, 11) and result.dtype == torch.float32
        assert torch.equal(hidden, source)
        retained = result.clone()
        changed = _rpu_convolution(module, hidden + 1, weight, stats)
        assert torch.equal(result, retained) and not torch.equal(changed, result)
        assert stats["calls"] == 2 and not torch.equal(*seen)


@pytest.mark.parametrize("dilation", [1, 3, 9])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_residual_host_im2col_matches_rounded_conv_across_core_budgets(monkeypatch, dilation, cores):
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
        self if a == ("rpu",) else original_to(self, *a, **kw))
    seen = []
    def linear(hidden, weight, bias, partition, active_cores, acc32):
        assert hidden.dtype == weight.dtype == torch.float16
        assert (bias, partition, active_cores, acc32) == (None, 1, cores, True)
        seen.append(hidden.clone())
        return (hidden.float() @ weight.float().T).half()
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", linear, raising=False)
    torch.manual_seed(1)
    module = torch.nn.Conv1d(96, 96, 7, dilation=dilation).eval()
    hidden = torch.randn(1, 96, 2 * (6 * dilation + 11))[:, :, ::2]
    weight = F.pad(module.weight.detach().flatten(1).half(), (0, 0, 0, 32))
    stats = {"calls": 0, "num_cores": cores}
    with torch.inference_mode():
        result = _residual_convolution(module, hidden, weight, stats)
        expected = F.conv1d(hidden.half().float(), module.weight.half().float(), dilation=dilation).half().float()
        expected += module.bias[None, :, None]
        torch.testing.assert_close(result, expected, rtol=0.002, atol=0.001)
        assert result.shape == (1, 96, 11) and result.dtype == torch.float32
        retained = result.clone()
        changed = _residual_convolution(module, hidden + 1, weight, stats)
        assert torch.equal(result, retained) and not torch.equal(changed, result)
        assert stats["calls"] == 2 and not torch.equal(*seen)


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_rpu_residual_install_leaves_non_residual_kernel7_layers_unchanged(monkeypatch, cores):
    from rpu_backend.runtime import weights
    decoder = torch.nn.Sequential(torch.nn.Conv1d(64, 1536, 7),
        *(torch.nn.Conv1d(c, c, 7, dilation=d) for c in (768, 384, 192, 96) for d in (1, 3, 9)),
        torch.nn.Conv1d(96, 1, 7), torch.nn.Conv1d(64, 64, 7, groups=64))
    originals = [m.forward for m in decoder]
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
        self if a == ("rpu",) else original_to(self, *a, **kw))
    converted = []
    def transform(weight, partition, active_cores):
        converted.append(weight)
        assert weight.shape[0] % (16 * cores) == 0 and (partition, active_cores) == (1, cores)
        return weight
    monkeypatch.setattr(weights, "transform_linear_weight", transform)
    restore, stats = [], {"calls": 0}
    install_rpu_residual_convolutions(decoder, restore, stats, num_cores=cores)
    assert stats["num_cores"] == cores
    assert stats["layers"] == len(restore) == len(converted) == 12
    for index in (0, 13, 14):
        assert decoder[index].forward == originals[index]
    for module, forward in reversed(restore):
        module.forward = forward
    assert [m.forward for m in decoder] == originals


def _decoder():
    decoder = torch.nn.Module()
    decoder.quantizer = torch.nn.ModuleDict({name: torch.nn.ModuleDict({
        "input_proj": torch.nn.Conv1d(512, 256, 1, bias=False),
        "output_proj": torch.nn.Conv1d(256, 512, 1, bias=False),
    }) for name in ("rvq_first", "rvq_rest")})
    decoder.convs = torch.nn.ModuleList([
        *(torch.nn.Conv1d(c, c, k, dilation=d) for c in (768, 384, 192, 96)
          for k, d in ((7, 1), (7, 3), (7, 9), (1, 1), (1, 1), (1, 1))),
        torch.nn.Conv1d(512, 1024, 3), torch.nn.Conv1d(1024, 1536, 7),
        torch.nn.Conv1d(1024, 1024, 7, groups=1024),
        torch.nn.Conv1d(1024, 1024, 7, groups=1024), torch.nn.Conv1d(96, 1, 7),
    ])
    decoder.transposes = torch.nn.ModuleList([
        torch.nn.ConvTranspose1d(1024, 1024, 2, stride=2),
        torch.nn.ConvTranspose1d(1024, 1024, 2, stride=2),
        torch.nn.ConvTranspose1d(1536, 768, 16, stride=8),
        torch.nn.ConvTranspose1d(768, 384, 10, stride=5),
        torch.nn.ConvTranspose1d(384, 192, 8, stride=4),
        torch.nn.ConvTranspose1d(192, 96, 6, stride=3),
    ])
    block_type = type("Qwen3TTSTokenizerV2ConvNeXtBlock", (torch.nn.Module,), {})
    decoder.convnext = torch.nn.ModuleList()
    for index in range(2):
        block = block_type()
        block.dwconv = torch.nn.Module()
        block.dwconv.conv = decoder.convs[-3 + index]
        block.norm = torch.nn.LayerNorm(1024, eps=1e-6)
        block.pwconv1 = torch.nn.Linear(1024, 4096)
        block.act = torch.nn.GELU()
        block.pwconv2 = torch.nn.Linear(4096, 1024)
        block.gamma = torch.nn.Parameter(torch.full((1024,), 1e-6))
        decoder.convnext.append(block)
    return decoder


def test_device_bias_overflow_rejects_before_transform(monkeypatch):
    from rpu_backend.runtime import weights
    decoder = _decoder()
    with torch.no_grad():
        decoder.convs[-1].bias.fill_(1e10)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv1d_spm_bias", lambda *a: None, raising=False)
    monkeypatch.setattr(weights, "transform_linear_weight", lambda *a: pytest.fail("premature transform"))
    restore = []
    with pytest.raises(ValueError, match="bias exceeds finite FP16"):
        install_rpu_convolutions(decoder, restore, {"calls": 0})
    assert not restore


@pytest.mark.parametrize("count", [0, 255 * 640 + 1])
def test_transpose_device_post_rejects_length_before_linear(monkeypatch, count):
    module = torch.nn.ConvTranspose1d(192, 96, 6, stride=3)
    hidden = torch.zeros(1, 192, 1).expand(1, 192, count)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation",
                        lambda *a: pytest.fail("premature linear"), raising=False)
    with pytest.raises(ValueError, match="sample count"):
        _rpu_transpose_convolution(module, hidden, None, {}, device_post=True)


def test_transpose_device_bias_overflow_rejects_before_transform(monkeypatch):
    from rpu_backend.runtime import weights
    decoder = _decoder()
    with torch.no_grad():
        decoder.transposes[-1].bias.fill_(1e10)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_transpose_overlap", lambda *a: None, raising=False)
    monkeypatch.setattr(weights, "transform_linear_weight", lambda *a: pytest.fail("premature transform"))
    restore = []
    with pytest.raises(ValueError, match="bias exceeds finite FP16"):
        install_rpu_convolutions(decoder, restore, {"calls": 0})
    assert not restore


def test_transpose_new_layout_and_device_post_are_bound_together(monkeypatch):
    from rpu_backend.runtime import weights
    decoder = _decoder()
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
        self if a == ("rpu",) else original_to(self, *a, **kw))
    monkeypatch.setattr(weights, "transform_linear_weight", lambda weight, *args: weight)
    monkeypatch.delattr(torch.ops.rpu, "tts_codec_conv_transpose1d_device_post", raising=False)
    def post(projected, bias, channels, kernel, stride):
        p = (channels + 127) // 128 * 128
        count = projected.shape[0]
        result = torch.zeros((count + (kernel > stride)) * stride, p, dtype=torch.float16)
        for index in range(count):
            result[index*stride:index*stride+kernel].add_(projected[index].view(kernel, p))
        return result + bias
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_transpose_overlap", post, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation",
        lambda x, w, *args: (x.float() @ w.float().T).half(), raising=False)
    restore, stats = [], {"calls": 0}
    install_rpu_convolutions(decoder, restore, stats)
    for module in decoder.transposes:
        packed, bias = module.forward.__func__.__defaults__[1:]
        k, c, cin = module.kernel_size[0], module.out_channels, module.in_channels
        natural = packed.view(k, -1, cin)
        assert torch.equal(natural[:, :c], module.weight.detach().permute(2, 1, 0).half())
        assert torch.count_nonzero(natural[:, c:]) == 0
        x = torch.randn(1, cin, 2)
        actual = module(x)
        expected = post((x.squeeze(0).T.half().contiguous().float() @ packed.float().T).half(),
                        bias, c, k, module.stride[0])[:, :c].T.unsqueeze(0).float()
        assert torch.equal(actual, expected)
    assert stats["transposed_calls"] == 6


@pytest.mark.parametrize("padded_rows", [False, True])
def test_resident_stage_keeps_three_units_on_device_and_restores(monkeypatch, padded_rows):
    from rpu_backend.runtime import weights
    decoder = _paired_decoder()
    for stage in decoder.decoder[1:]:
        for unit in stage.block[2:]:
            channels = unit.conv1.conv.in_channels
            unit.act1 = SnakeBeta()
            unit.act1.alpha = torch.nn.Parameter(torch.zeros(channels))
            unit.act1.beta = torch.nn.Parameter(torch.zeros(channels))
    original = {m: m.forward for m in decoder.modules()}
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
        self if a == ("rpu",) else original_to(self, *a, **kw))
    monkeypatch.setattr(weights, "transform_linear_weight", lambda value, *args: value)
    monkeypatch.delattr(torch.ops.rpu, "tts_codec_conv1d_spm", raising=False)
    for name in ("tts_codec_residual_unit", "tts_codec_residual_unit_complete"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *a: pytest.fail("CPU boundary used"), raising=False)
    seen = []
    def native(rows, first, affine, second, dilation, channels):
        assert rows.dtype == torch.float16 and rows.is_contiguous()
        physical = (channels + 127) // 128 * 128
        assert affine.shape == (6, physical)
        assert first.shape == (physical, 7 * physical)
        if channels != physical:
            assert torch.count_nonzero(rows[:, channels:]) == 0
            assert torch.count_nonzero(first.view(physical, 7, physical)[:, :, channels:]) == 0
            assert torch.count_nonzero(first[channels:]) == 0
        seen.append((channels, dilation))
        result = rows + dilation
        result[:, channels:] = 0
        return result
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_residual_unit_rows", native, raising=False)
    if padded_rows:
        monkeypatch.setattr(torch.ops.rpu, "tts_codec_residual_unit_padded_rows", native, raising=False)
    else:
        monkeypatch.delattr(torch.ops.rpu, "tts_codec_residual_unit_padded_rows", raising=False)
    restore, stats = [], {"calls": 0}
    install_rpu_convolutions(decoder, restore, stats)
    stages = [(1, 768), (2, 384)] + ([(3, 192), (4, 96)] if padded_rows else [])
    for index, channels in stages:
        hidden = torch.randn(1, channels, 7)[..., 1::2]
        expected = ((hidden.half() + 1) + 3) + 9
        assert torch.equal(decoder.decoder[index](hidden), expected.float())
    assert seen == [(c, d) for _, c in stages for d in (1, 3, 9)]
    assert stats["paired_calls"] == 3 * len(stages) and stats["calls"] == 6 * len(stages)
    if not padded_rows:
        assert decoder.decoder[3].forward == original[decoder.decoder[3]]
    for module, forward in reversed(restore):
        module.forward = forward
    assert all(module.forward == forward for module, forward in original.items())


@pytest.mark.parametrize("index", [3, 4])
def test_padded_resident_stage_rejects_extra_blocks_before_transform(monkeypatch, index):
    from rpu_backend.adapters.qwen3_tts import codec
    from rpu_backend.runtime import weights
    decoder = _paired_decoder()
    decoder.decoder[index].block.append(torch.nn.Identity())
    for name in ("tts_codec_residual_unit", "tts_codec_residual_unit_complete",
                 "tts_codec_residual_unit_rows", "tts_codec_residual_unit_padded_rows"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *args: None, raising=False)
    monkeypatch.setattr(codec, "_residual_pair_affines", lambda *args, **kwargs: [])
    monkeypatch.setattr(weights, "transform_linear_weight", lambda *args: pytest.fail("transformed before validation"))
    restore = []
    with pytest.raises(ValueError, match="exactly three units"):
        install_rpu_convolutions(decoder, restore, {"calls": 0})
    assert not restore


def test_rpu_install_validates_all_layers_before_transform(monkeypatch):
    from rpu_backend.runtime import weights
    decoder = _decoder()
    decoder.convs[-1].padding = (1,)
    originals = [m.forward for m in decoder.modules()]
    monkeypatch.setattr(weights, "transform_linear_weight", lambda *a: pytest.fail("transformed before validation"))
    restore = []
    with pytest.raises(ValueError, match="geometry"):
        install_rpu_convolutions(decoder, restore, {"calls": 0})
    assert restore == [] and [m.forward for m in decoder.modules()] == originals


def test_rpu_install_rejects_convnext_profile_before_transform(monkeypatch):
    from rpu_backend.runtime import weights
    decoder = _decoder()
    decoder.convnext[1].pwconv2 = torch.nn.Linear(4096, 1000)
    monkeypatch.setattr(weights, "transform_linear_weight",
                        lambda *args: pytest.fail("transformed before validation"))
    restore = []
    with pytest.raises(ValueError, match="ConvNeXt Linear geometry"):
        install_rpu_convolutions(decoder, restore, {"calls": 0})
    assert not restore


def test_invalid_codec_backend_rejected_before_loading(monkeypatch):
    from rpu_backend.api import Qwen3TTSModel
    from rpu_backend.adapters import qwen3_tts
    monkeypatch.setattr(qwen3_tts, "load_cpu_model", lambda *a: pytest.fail("weights loaded"))
    with pytest.raises(ValueError, match="codec_backend"):
        Qwen3TTSModel.from_pretrained("unused", source="unused", codec_backend="cuda")


@pytest.mark.parametrize("budget", [4, 6, 8])
def test_rpu_install_covers_decoding_convs_and_preserves_unused_layers(monkeypatch, budget):
    from rpu_backend.runtime import weights
    expected_cores = 8 if budget == 8 else 4
    decoder = _decoder()
    originals = {m: m.forward for m in decoder.modules()}
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
        self if a == ("rpu",) else original_to(self, *a, **kw))
    for name in ("tts_codec_conv1d_spm", "tts_codec_conv1d_spm_bias", "tts_codec_convnext_block",
                 "tts_codec_snake", "tts_codec_residual_unit", "tts_codec_conv_pair",
                 "tts_codec_transpose_overlap"):
        monkeypatch.delattr(torch.ops.rpu, name, raising=False)
    converted = []
    def transform(weight, partition, cores):
        converted.append(weight.shape)
        assert weight.shape[0] % 128 == 0 and (partition, cores) == (1, expected_cores)
        return weight
    monkeypatch.setattr(weights, "transform_linear_weight", transform)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv_transpose1d", lambda *args: None,
                        raising=False)
    restore, stats = [], {"calls": 0}
    install_rpu_convolutions(decoder, restore, stats, num_cores=budget)
    assert stats["num_cores"] == expected_cores
    assert stats["layers"] == 31
    assert stats["transposed_layers"] == 6 and stats["transposed_calls"] == 0
    assert stats["convnext_linears"] == 4 and stats["convnext_linear_calls"] == 0
    assert len(restore) == len(converted) == 41
    for module in (decoder.quantizer.rvq_first.input_proj,
                   decoder.quantizer.rvq_rest.input_proj):
        assert module.forward == originals[module]
    assert all(m.weight.device.type == "cpu" and m.weight.dtype == torch.float32 for m, _ in restore)
    # The diagonal expansion is checked using the actual prepared forward weight.
    for module in (decoder.convs[-3], decoder.convs[-2]):
        weight = module.forward.__func__.__defaults__[0].view(1024, 1024, 7)
        channel = torch.arange(1024)
        assert torch.equal(weight[channel, channel], module.weight[:, 0].half())
        assert torch.count_nonzero(weight) == torch.count_nonzero(module.weight.half())
    for module, forward in reversed(restore):
        module.forward = forward
    assert all(m.forward == forward for m, forward in originals.items())


def test_convnext_linear_uses_acc32_and_returns_owned_cpu_fp32(monkeypatch):
    module = torch.nn.Linear(4, 8).eval()
    hidden = torch.randn(1, 3, 4)
    source = hidden.clone()
    weight, bias = module.weight.detach().half(), module.bias.detach().half()
    stats = {"convnext_linear_calls": 0}
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
                        self if a == ("rpu",) else original_to(self, *a, **kw))
    seen = []

    def linear(value, packed, device_bias, partition, cores, acc32):
        assert value.dtype == packed.dtype == device_bias.dtype == torch.float16
        assert (partition, cores, acc32) == (1, 8, True)
        seen.append(value.clone())
        return (value.float() @ packed.float().T + device_bias.float()).half()

    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", linear, raising=False)
    result = _rpu_linear(module, hidden, weight, bias, stats)
    expected = (hidden.half().float() @ weight.float().T + bias.float()).half().float()
    assert torch.equal(result, expected) and result.dtype == torch.float32
    assert torch.equal(hidden, source) and stats["convnext_linear_calls"] == 1
    retained = result.clone()
    changed = _rpu_linear(module, hidden + 1, weight, bias, stats)
    assert torch.equal(result, retained) and not torch.equal(changed, result)
    assert stats["convnext_linear_calls"] == 2 and not torch.equal(*seen)
    with pytest.raises(ValueError, match="CPU FP32"):
        _rpu_linear(module, hidden.half(), weight, bias, stats)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("linear failure")))
    with pytest.raises(RuntimeError, match="linear failure"):
        _rpu_linear(module, hidden, weight, bias, stats)
    assert stats["convnext_linear_calls"] == 2


@pytest.mark.parametrize("kernel,stride", [(2, 2), (16, 8)])
def test_native_transpose_transport_preserves_arguments_and_errors(monkeypatch, kernel, stride):
    module = torch.nn.ConvTranspose1d(8, 5, kernel, stride=stride).eval()
    hidden = torch.randn(1, 8, 7)
    weight = object()
    output = torch.randn(1, 5, (hidden.shape[-1] - 1) * stride + kernel)
    stats = {"transposed_calls": 0}

    def native(value, packed, bias, channels, native_kernel, native_stride):
        assert value is hidden and packed is weight and bias is module.bias
        assert (channels, native_kernel, native_stride) == (5, kernel, stride)
        return output

    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv_transpose1d", native, raising=False)
    assert _rpu_transpose_convolution(module, hidden, weight, stats) is output
    assert stats["transposed_calls"] == 1
    with pytest.raises(ValueError, match="CPU FP32"):
        _rpu_transpose_convolution(module, hidden.half(), weight, stats)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv_transpose1d",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("device failure")))
    with pytest.raises(RuntimeError, match="device failure"):
        _rpu_transpose_convolution(module, hidden, weight, stats)
    assert stats["transposed_calls"] == 1


@pytest.mark.parametrize("channels,kernel,stride", [(1024, 2, 2), (768, 16, 8),
                                                  (384, 10, 5), (192, 8, 4), (96, 6, 3)])
def test_device_post_native_boundary_preserves_arguments(monkeypatch, channels, kernel, stride):
    in_channels = 1024 if channels == 1024 else channels * 2
    module = torch.nn.ConvTranspose1d(in_channels, channels, kernel, stride=stride)
    hidden = torch.zeros(1, in_channels, 6)[..., ::2]
    weight, bias, output = object(), object(), object()
    def native(*args):
        assert args == (hidden, weight, bias, channels, kernel, stride)
        return output
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv_transpose1d_device_post", native, raising=False)
    stats = {"transposed_calls": 0}
    assert _rpu_transpose_convolution(module, hidden, weight, stats, device_post=True, bias=bias) is output
    assert stats["transposed_calls"] == 1
    with pytest.raises(ValueError, match="CPU FP32"):
        _rpu_transpose_convolution(module, hidden.half(), weight, stats)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv_transpose1d_device_post",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("device failure")))
    with pytest.raises(RuntimeError, match="device failure"):
        _rpu_transpose_convolution(module, hidden, weight, stats, device_post=True, bias=bias)
    assert stats["transposed_calls"] == 1


def test_native_codec_transport_preserves_arguments_and_propagates_errors(monkeypatch):
    module = torch.nn.Conv1d(96, 96, 7, dilation=3).eval()
    hidden = torch.randn(1, 96, 50)[:, :, ::2]
    weight = object()
    output = torch.randn(1, 96, 7)
    stats = {"calls": 0}
    monkeypatch.delattr(torch.ops.rpu, "tts_codec_conv1d_spm", raising=False)
    monkeypatch.delattr(torch.ops.rpu, "tts_codec_conv1d_spm_bias", raising=False)
    def native(value, packed, bias, kernel, dilation, channels):
        assert value is hidden and packed is weight and bias is module.bias
        assert (kernel, dilation, channels) == (7, 3, 96)
        return output
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv1d", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation",
                        lambda *args: pytest.fail("staged path executed"), raising=False)
    assert _rpu_convolution(module, hidden, weight, stats) is output and stats["calls"] == 1
    def fail(*args):
        raise RuntimeError("device failure")
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv1d", fail)
    with pytest.raises(RuntimeError, match="device failure"):
        _rpu_convolution(module, hidden, weight, stats)
    assert stats["calls"] == 1
    with pytest.raises(ValueError, match="CPU FP32"):
        _rpu_convolution(module, hidden.half(), weight, stats)


def _paired_decoder():
    decoder = _decoder()
    residual_type = type("Qwen3TTSTokenizerV2DecoderDecoderResidualUnit", (torch.nn.Module,), {})
    stages = [torch.nn.Identity()]
    for stage, channels in enumerate((768, 384, 192, 96)):
        blocks = [torch.nn.Identity(), torch.nn.Identity()]
        for index, dilation in enumerate((1, 3, 9)):
            unit = residual_type()
            unit.act1 = torch.nn.Identity()
            unit.act2 = SnakeBeta()
            unit.act2.alpha = torch.nn.Parameter(torch.zeros(channels))
            unit.act2.beta = torch.nn.Parameter(torch.zeros(channels))
            unit.conv1, unit.conv2 = torch.nn.Module(), torch.nn.Module()
            unit.conv1.conv = decoder.convs[stage * 6 + index]
            unit.conv2.conv = decoder.convs[stage * 6 + index + 3]
            unit.conv1.padding, unit.conv1.dilation = 6 * dilation, dilation
            unit.conv2.padding = 0
            blocks.append(unit)
        group = torch.nn.Module()
        group.block = torch.nn.ModuleList(blocks)
        stages.append(group)
    decoder.convs = decoder.convs[24:]
    decoder.decoder = torch.nn.ModuleList(stages)
    return decoder


def test_residual_pair_binding_padding_calls_errors_and_restore(monkeypatch):
    from rpu_backend.runtime import weights
    decoder = _paired_decoder()
    originals = {m: m.forward for m in decoder.modules()}
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
                        self if a == ("rpu",) else original_to(self, *a, **kw))
    monkeypatch.setattr(weights, "transform_linear_weight", lambda weight, *args: weight)
    for name in ("tts_codec_conv1d_spm", "tts_codec_conv1d_spm_bias", "tts_codec_residual_unit",
                 "tts_codec_residual_unit_complete", "tts_codec_residual_unit_rows",
                 "tts_codec_residual_unit_padded_rows", "tts_codec_snake"):
        monkeypatch.delattr(torch.ops.rpu, name, raising=False)
    seen = []
    def pair(hidden, first, affine, second, bias, dilation, channels):
        n = (channels + 127) // 128 * 128
        assert first.shape == (n, channels * 7) and second.shape == (n, n)
        assert affine.shape == (3, n) and affine.dtype == torch.float16
        assert torch.equal(affine[1, channels:], torch.ones(n - channels))
        assert torch.count_nonzero(affine[0, channels:]) == torch.count_nonzero(affine[2, channels:]) == 0
        assert not torch.count_nonzero(second[channels:]) and not torch.count_nonzero(second[:, channels:])
        seen.append(hidden.clone())
        return torch.zeros(1, channels, hidden.shape[-1] - 6 * dilation)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv_pair", pair, raising=False)
    restore, stats = [], {"calls": 0}
    install_rpu_convolutions(decoder, restore, stats)
    assert stats["residual_pairs"] == 12 and stats["paired_calls"] == 0
    unit = decoder.decoder[4].block[3]
    hidden = torch.randn(1, 96, 17)
    result = unit(hidden)
    assert torch.equal(result, hidden) and result.data_ptr() != hidden.data_ptr()
    assert torch.equal(seen[0], F.pad(hidden, (18, 0)))
    assert stats["calls"] == 2 and stats["paired_calls"] == 1
    def fail(*args):
        raise RuntimeError("pair failure")
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv_pair", fail)
    with pytest.raises(RuntimeError, match="pair failure"):
        unit(hidden)
    assert stats["calls"] == 2 and stats["paired_calls"] == 1
    with pytest.raises(ValueError, match="CPU FP32"):
        unit(hidden.half())
    for module, original in reversed(restore):
        module.forward = original
    assert all(m.forward == forward for m, forward in originals.items())


@pytest.mark.parametrize("complete", [False, True])
def test_spm_codec_binds_full_residual_and_convnext_blocks(monkeypatch, complete):
    from rpu_backend.runtime import weights
    decoder = _paired_decoder()
    for stage, channels in enumerate((1536, 768, 384, 192), start=1):
        snake = SnakeBeta()
        snake.alpha = torch.nn.Parameter(torch.zeros(channels))
        snake.beta = torch.nn.Parameter(torch.zeros(channels))
        decoder.decoder[stage].block[0] = snake
    final_snake = SnakeBeta()
    final_snake.alpha = torch.nn.Parameter(torch.zeros(96))
    final_snake.beta = torch.nn.Parameter(torch.zeros(96))
    decoder.decoder.append(final_snake)
    for unit in (module for module in decoder.modules()
                 if type(module).__name__ == "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit"):
        unit.act1 = SnakeBeta()
        unit.act1.alpha = torch.nn.Parameter(torch.zeros(unit.conv1.conv.in_channels))
        unit.act1.beta = torch.nn.Parameter(torch.zeros(unit.conv1.conv.in_channels))
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
                        self if a == ("rpu",) else original_to(self, *a, **kw))
    monkeypatch.setattr(weights, "transform_linear_weight", lambda weight, *args: weight)
    for name in ("tts_codec_residual_unit_rows", "tts_codec_residual_unit_padded_rows",
                 "tts_codec_residual_unit_complete"):
        monkeypatch.delattr(torch.ops.rpu, name, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv1d_spm", lambda *args: args[0], raising=False)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv_transpose1d", lambda *args: args[0], raising=False)
    residual_calls, convnext_calls, snake_calls = [], [], []
    def residual(hidden, first, affine, second, dilation, channels):
        padded = (channels + 127) // 128 * 128
        assert first.shape == (padded, channels * 7)
        assert second.shape == (padded, padded)
        assert affine.shape == (6 if complete else 4, padded) and affine.dtype == torch.float16
        residual_calls.append((dilation, channels))
        return hidden.clone() if complete else torch.zeros_like(hidden)
    def convnext(hidden, depthwise, depthwise_bias, norm_weight, norm_bias,
                 first, first_bias, second, second_bias, gamma, eps):
        assert depthwise.shape == (1024, 7 * 1024)
        assert first.shape == (4096, 1024) and second.shape == (1024, 4096)
        assert all(value.dtype == torch.float16 for value in
                   (depthwise_bias, norm_weight, norm_bias, first_bias,
                    second_bias, gamma))
        assert eps == 1e-6
        convnext_calls.append(hidden.shape)
        return hidden.clone()
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_residual_unit", residual, raising=False)
    if complete:
        monkeypatch.setattr(torch.ops.rpu, "tts_codec_residual_unit_complete", residual, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_convnext_block", convnext, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_snake",
                        lambda hidden, affine, channels:
                        snake_calls.append((affine.shape, channels)) or hidden.clone(),
                        raising=False)
    restore, stats = [], {"calls": 0}
    if complete:
        alpha = decoder.decoder[4].block[3].act1.alpha
        with torch.no_grad():
            alpha.fill_(1000)
        with pytest.raises(ValueError, match="finite FP16"):
            install_rpu_convolutions(decoder, restore, stats)
        assert not restore
        with torch.no_grad():
            alpha.zero_()
    install_rpu_convolutions(decoder, restore, stats)
    assert stats["transport"] == "spm_im2col"
    assert stats["residual_pairs"] == 12 and stats["convnext_blocks"] == 2
    assert stats["standalone_snakes"] == 5
    hidden = torch.randn(1, 96, 17)
    assert torch.equal(decoder.decoder[4].block[3](hidden), hidden)
    assert residual_calls == [(3, 96)] and stats["paired_calls"] == 1
    block_hidden = torch.randn(1, 1024, 9)
    assert torch.equal(decoder.convnext[0](block_hidden), block_hidden)
    assert convnext_calls == [block_hidden.shape] and stats["convnext_block_calls"] == 1
    assert torch.equal(decoder.decoder[5](hidden), hidden)
    assert snake_calls == [((2, 128), 96)] and stats["standalone_snake_calls"] == 1


@pytest.mark.parametrize("invalid", ["geometry", "child_dilation", "range"])
def test_residual_pair_rejects_before_transform(monkeypatch, invalid):
    from rpu_backend.runtime import weights
    decoder = _paired_decoder()
    unit = decoder.decoder[1].block[2]
    if invalid == "geometry":
        unit.conv1.padding = 1
    elif invalid == "child_dilation":
        unit.conv1.conv.dilation = (9,)
        decoder.decoder[1].block[4].conv1.conv.dilation = (1,)
    else:
        unit.act2.alpha.data.fill_(100)
    for name in ("tts_codec_residual_unit", "tts_codec_residual_unit_complete"):
        monkeypatch.delattr(torch.ops.rpu, name, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "tts_codec_conv_pair", lambda *args: None, raising=False)
    monkeypatch.setattr(weights, "transform_linear_weight", lambda *args: pytest.fail("premature transform"))
    restore = []
    with pytest.raises(ValueError, match="geometry|finite FP16"):
        install_rpu_convolutions(decoder, restore, {"calls": 0})
    assert not restore
