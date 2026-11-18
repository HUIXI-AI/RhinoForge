"""Frozen CPU codec geometry, overlap boundaries, and output ownership."""
import pytest
import torch
import torch.nn.functional as F

from rpu_backend.adapters.qwen3_tts.codec import (
    install_codec_convolutions, install_rpu_residual_convolutions, _residual_convolution,
)


@pytest.mark.parametrize("kernel,stride", [(2, 2), (16, 8), (10, 5), (8, 4), (6, 3)])
@pytest.mark.parametrize("length", [1, 7])
@pytest.mark.parametrize("bias", [False, True])
def test_transpose_matches_previous_fold_at_boundaries(kernel, stride, length, bias):
    torch.manual_seed(12)
    module = torch.nn.ConvTranspose1d(3, 5, kernel, stride=stride, bias=bias).eval()
    hidden = torch.randn(1, 3, length * 2)[:, :, ::2]
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
    original = module.forward
    restore = []
    with torch.inference_mode():
        expected = module(hidden)
        install_codec_convolutions(module, restore)
        result = module(hidden)
        assert result.shape == expected.shape and result.dtype == torch.float32
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, expected)
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


@pytest.mark.parametrize("dilation", [1, 3, 9])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_rpu_im2col_matches_rounded_conv_and_owns_outputs(monkeypatch, dilation, cores):
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


def test_rpu_install_validates_all_layers_before_transform(monkeypatch):
    from rpu_backend.runtime import weights
    decoder = torch.nn.Sequential(*(torch.nn.Conv1d(c, c, 7, dilation=d)
        for c in (768, 384, 192, 96) for d in (1, 3, 9)))
    decoder[-1].padding = (1,)
    originals = [m.forward for m in decoder]
    monkeypatch.setattr(weights, "transform_linear_weight", lambda *a: pytest.fail("transformed before validation"))
    restore = []
    with pytest.raises(ValueError, match="geometry"):
        install_rpu_residual_convolutions(decoder, restore, {"calls": 0})
    assert restore == [] and [m.forward for m in decoder] == originals


def test_invalid_codec_backend_rejected_before_loading(monkeypatch):
    from rpu_backend.api import Qwen3TTSModel
    from rpu_backend.adapters import qwen3_tts
    monkeypatch.setattr(qwen3_tts, "load_cpu_model", lambda *a: pytest.fail("weights loaded"))
    with pytest.raises(ValueError, match="codec_backend"):
        Qwen3TTSModel.from_pretrained("unused", source="unused", codec_backend="cuda")


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_rpu_install_leaves_non_residual_kernel7_layers_unchanged(monkeypatch, cores):
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
