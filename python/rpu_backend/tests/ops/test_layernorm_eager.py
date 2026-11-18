import os
import pytest
import torch

@pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware (set RPU_HW_AVAILABLE=1)')
def test_layernorm_eager_nonzero_matches_cpu(record_property):
    import rpu_backend  # noqa: F401  (registers torch.rpu)
    from rpu_backend.tests.conformance._numeric import format_metrics, numeric_metrics

    generator = torch.Generator().manual_seed(42)
    N = 17
    eps = 1e-5
    # C=768 -> LAYER_NORM (C>512), C=256 -> LAYER_NORM_SIMPLE (C<=512): both share the
    # same high-range reg ABI, so cover both kernels. reg21=has_affine is hardwired to 1
    # because rpu_layernorm .value()s weight+bias (a no-affine layer_norm throws before
    # the launcher), so affine is always present here.
    # Exercise both sides of the dispatch boundary and an incomplete row block.
    # The updated BASE kernels fix tail clearing; the register ABI is unchanged.
    # C=121/513 also cover non-bank-sized affine/input/output SPM transfers.
    for C in (121, 256, 512, 513, 768):
        x = torch.randn(N, C, dtype=torch.float16, generator=generator)
        w = torch.randn(C, dtype=torch.float16, generator=generator)
        b = torch.randn(C, dtype=torch.float16, generator=generator)

        ref = torch.layer_norm(x.float(), (C,), w.float(), b.float(), eps)
        got = torch.layer_norm(x.to("rpu"), (C,), w.to("rpu"), b.to("rpu"), eps).cpu()

        metrics = numeric_metrics(got, ref, actual_dtype=torch.float16)
        record_property(f"layer_norm_C{C}_numeric", metrics)
        assert got.abs().sum().item() > 0, f"RPU eager layer_norm C={C} collapsed to all-zeros (reg ABI bug)"
        print(f"[PASS] eager layer_norm C={C}: nonzero; CPU fp32 diagnostics: "
              f"{format_metrics(metrics)}")

@pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware (set RPU_HW_AVAILABLE=1)')
def test_eager_mean_last_dim_matches_fp32(record_property):
    import rpu_backend
    from rpu_backend.tests._seed import input_generator
    from rpu_backend.tests.conformance._numeric import require_same_semantic
    generator = input_generator()
    for shape in ((16, 32), (2, 16, 256)):
        x = (0.25 + 1.75 * torch.rand(shape, generator=generator)).half()
        for keepdim in (False, True):
            ref = torch.mean(x.float(), dim=-1, keepdim=keepdim)
            got = torch.mean(x.to('rpu'), dim=-1, keepdim=keepdim).cpu()
            require_same_semantic(got, ref, actual_dtype=torch.float16, cosine_min=0.999, relative_l2_max=0.01, row_relative_l2_max=0.01, label=f'eager mean FP16 {shape} keepdim={keepdim}')
if __name__ == '__main__':
    test_layernorm_eager_nonzero_matches_cpu(lambda *_: None)
    test_eager_mean_last_dim_matches_fp32(lambda *_: None)
