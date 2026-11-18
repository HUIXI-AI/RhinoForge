"""Production Dense W4: unified I4 G32 / U4 G128, BUILD + three REPLAYs.

Uses normal librpu_backend bindings; BUILD_CONFORMANCE is not required.
Run only under the ordinary exclusive-device supervisor with RPU_HW_AVAILABLE=1.
The default geometry is a real 2B gate/up or down projection. Override hidden
and intermediate dimensions for additional model shapes without changing q/z/s.
"""
import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RPU_HW_AVAILABLE") != "1", reason="requires exclusive device")


@pytest.mark.parametrize("acc32", [False, True], ids=["acc16", "acc32"])
@pytest.mark.parametrize("unsigned,group_size", [(False, 32), (True, 128)])
@pytest.mark.parametrize("partition", [1, 0], ids=["gate_up", "down"])
@pytest.mark.parametrize("m", [1, 32], ids=["decode", "prefill"])
def test_shared_w4_production_graph(unsigned, group_size, partition, m, acc32):
    import rpu_backend  # noqa: F401
    from rpu_backend.graph import GraphCache, GraphSignature
    from rpu_backend.quant.int4_pgrp_pack import (
        swizzle_pack_int4_pgrp, swizzle_int4_pgrp_scale, w4_storage_shape)
    from rpu_backend.quant.awq_gemm import swizzle_pack_uint4_pgrp
    from rpu_backend.runtime import control

    control.set_caching_allocator(True)
    control.set_ddr_flush(True)
    control.set_ddr_flush_force(True)
    cores = 8
    hidden = int(os.environ.get("RPU_W4_PROBE_HIDDEN", "2048"))
    intermediate = int(os.environ.get("RPU_W4_PROBE_INTERMEDIATE", "6144"))
    n, k = (intermediate, hidden) if partition else (hidden, intermediate)
    local_n, local_k = (n // cores, k) if partition else (n, k // cores)
    generator = torch.Generator().manual_seed(3500 + m + partition + group_size)
    q = torch.randint(0 if unsigned else -8, 16 if unsigned else 8,
                      (n, k), generator=generator, dtype=torch.int8)
    scale = (torch.rand(k // group_size, n, generator=generator) * .007 + .001).half()
    zero = torch.randint(2, 14, scale.shape, generator=generator).half()
    if unsigned:
        assert (zero != 8).any(), "the probe must exercise asymmetric zeros"
    else:
        zero.zero_()
    # Exercise the actual public UINT4 packer as well as the signed
    # packer. Neither path re-quantizes or clips q-z.
    packed = (swizzle_pack_uint4_pgrp(q.to(torch.uint8), partition, cores)
              if unsigned else swizzle_pack_int4_pgrp(q, partition, cores))
    weight = packed.reshape(w4_storage_shape(n, k, partition, cores)).to("rpu")
    packed_scale = swizzle_int4_pgrp_scale(scale, group_size, partition, cores).to("rpu")
    packed_scale = torch.ops.rpu.w4_pgrp_prepare_metadata(weight, packed_scale)
    packed_zero = None
    if unsigned:
        packed_zero = swizzle_int4_pgrp_scale(-zero, group_size, partition, cores).to("rpu")
        packed_zero = torch.ops.rpu.w4_pgrp_prepare_metadata(weight, packed_zero)
    x = (torch.randn(cores, m, local_k, generator=generator) * .1).half()
    inputs = x.to("rpu")
    outputs = torch.empty(cores, m, local_n, dtype=torch.float16, device="rpu")
    cache = GraphCache(max_entries=1)
    signature = GraphSignature(op_id="shared_w4_probe", shapes=[cores, m, n, k],
                               dyn_dims=[partition, group_size, int(unsigned)],
                               dtypes=[torch.float16])
    observed = []
    try:
        for iteration, factor in enumerate((1.0, .5, -.5, 1.0)):
            current = (x * factor).half()
            inputs.copy_(current)
            with cache.capture(signature):
                torch.ops.rpu.parallel_linear_w4_test_out(
                    inputs, weight, packed_scale, packed_zero, outputs,
                    bool(partition), acc32)
            actual = outputs.cpu().float()
            assert actual.shape == (cores, m, local_n) and torch.isfinite(actual).all()
            observed.append(actual.clone())
            if iteration == 0:
                cache.freeze()
        assert torch.equal(observed[0], observed[3]), "same-input REPLAY drifted"
        assert not torch.equal(observed[0], observed[1]), "changed input was ignored"
        assert cache.size() == 1 and cache.cache_invariant_ok()
        entry, = cache.snapshot()
        assert entry.kernel_count > 0 and entry.replay_count == 3
        assert entry.recapture_count == 0
    finally:
        # Retire the recorded address owners before any input/weight is released.
        cache.clear()
