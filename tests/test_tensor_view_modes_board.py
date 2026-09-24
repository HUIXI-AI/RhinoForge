"""Explicit board regressions for persistent tensors crossing inference modes.

Views must retain the base's inference state, storage and version ownership.
These are metadata operations; no model or generated operator asset is needed.
"""
import os

import pytest
import rpu_backend  # noqa: F401
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RHINOFORGE_RUN_BOARD_TESTS") != "1"
    or not torch.rpu.is_available(),
    reason="requires explicit RHINOFORGE_RUN_BOARD_TESTS=1 and an idle, locked RPU",
)


def _operation(name, tensor):
    if name == "squeeze":
        return tensor.squeeze()
    if name == "squeeze_dim":
        return tensor.squeeze(0)
    if name == "unsqueeze":
        return tensor.unsqueeze(1)
    if name == "transpose":
        return tensor.transpose(1, 2)
    if name == "expand":
        return tensor.expand(4, 2, 3)
    if name == "as_strided":
        return tensor.as_strided((2, 3), (3, 1))
    if name == "view":
        return tensor.view(2, 3)
    raise AssertionError(name)


@pytest.mark.parametrize("base_inference", [False, True])
@pytest.mark.parametrize("view_inference", [False, True])
@pytest.mark.parametrize("operation", [
    "squeeze", "squeeze_dim", "unsqueeze", "transpose", "expand",
    "as_strided", "view",
])
def test_persistent_views_match_cpu_across_inference_modes(
    base_inference, view_inference, operation,
):
    with torch.no_grad(), torch.inference_mode(base_inference):
        backing = torch.arange(16, dtype=torch.float16)
        device_backing = backing.to("rpu")
        cpu = backing.as_strided((1, 2, 3), (6, 3, 1), storage_offset=4)
        device = device_backing.as_strided((1, 2, 3), (6, 3, 1), storage_offset=4)

    with torch.no_grad(), torch.inference_mode(view_inference):
        expected = _operation(operation, cpu)
        actual = _operation(operation, device)

    assert actual.shape == expected.shape
    assert actual.is_inference() == expected.is_inference() == base_inference
    assert actual.storage_offset() == expected.storage_offset() == 4
    assert actual.untyped_storage().data_ptr() == device.untyped_storage().data_ptr()
    # This backend exports .cpu() as a shared-DDR view; snapshot it explicitly.
    retained = actual.cpu().clone()
    assert torch.equal(retained, expected)
    if operation == "unsqueeze":
        # Preserve the backend's existing singleton stride (3 rather than 6).
        # Singleton strides do not alter addressing and are outside this fix.
        assert actual.stride() == (6, 3, 3, 1)
    else:
        assert actual.stride() == expected.stride()

    if not base_inference:
        version = device._version
        assert actual._version == version
    # A shared view observes a later producer update; an independent retained
    # snapshot does not. Updating under inference mode is valid for both bases.
    with torch.no_grad(), torch.inference_mode():
        device.copy_(torch.full((1, 2, 3), 19.0, dtype=torch.float16))
    assert torch.equal(actual.cpu(), torch.full_like(expected, 19.0))
    assert torch.equal(retained, expected)
    if not base_inference:
        assert device._version > version
        assert actual._version == device._version


@pytest.mark.parametrize("shape,dim", [((1, 1), None), ((1,), 0), ((), 0), ((), -1)])
def test_squeeze_all_singletons_returns_scalar(shape, dim):
    cpu = torch.ones(shape, dtype=torch.float16)
    device = cpu.to("rpu")
    actual = device.squeeze() if dim is None else device.squeeze(dim)
    expected = cpu.squeeze() if dim is None else cpu.squeeze(dim)
    assert actual.shape == expected.shape == torch.Size([])
    assert actual.data_ptr() == device.data_ptr()
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("dim", [-2, 1])
def test_squeeze_dim_rejects_out_of_range(dim):
    cpu = torch.ones(1, dtype=torch.float16)
    device = cpu.to("rpu")
    with pytest.raises(IndexError, match="Dimension out of range"):
        cpu.squeeze(dim)
    with pytest.raises(IndexError, match="Dimension out of range"):
        device.squeeze(dim)
