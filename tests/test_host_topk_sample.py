"""Execute the production top-k sampling helper without loading the backend.

The helper replaces a temperature/top-k/softmax/multinomial chain, so what is
worth checking is not one drawn token but that the *distribution* it draws from
is the chain's: the surviving candidate set, including the reference's tie rule,
and the probability mass inside it. Batched identical rows check both at once
and also check that the generator advances per row, because a sampler that
reuses one uniform passes every single-row test.

On a loaded backend, reuse ``check_host_topk_sample(...)`` with
``torch.ops.rpu.sample_topk_lastdim_host`` to put the same cases through the
real dispatcher.
"""
from pathlib import Path
import shutil
import subprocess

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TEMPERATURE, TOP_K = 0.9, 50


def reference_probs(logits, temperature=TEMPERATURE, top_k=TOP_K):
    """The chain the helper replaces, as transformers spells it."""
    from transformers.generation.logits_process import TopKLogitsWarper

    return TopKLogitsWarper(top_k)(None, logits.float() / temperature).softmax(-1)


def host_topk_sample_cases():
    """Rows whose candidate set is interesting: ties, plateaus, tiny support."""
    for dtype in (torch.float32, torch.float16):
        for width in (1, 11, 51, 64, 128, 2048):
            x = torch.randn((9, width), generator=torch.Generator().manual_seed(width),
                            dtype=dtype) * 4
            # A flat row ties every column with the k-th largest, so the
            # reference keeps all of them where a fixed-size set would not.
            x[1].fill_(1.5)
            # Exactly one candidate above a flat floor.
            x[2].fill_(-8)
            x[2, min(7, width - 1)] = 9
            # A plateau straddling the k-th place.
            x[3].fill_(-3)
            x[3, : min(TOP_K + 13, width)] = 2.25
            # Signed zeros: -0.0 is not below 0.0, so both survive.
            x[4].zero_()
            x[4, ::2] = -0.0
            # Large magnitudes: the shared max subtraction must keep exp finite,
            # and the floor must still round to exactly zero probability.
            x[5].fill_(-60000 if dtype is torch.float16 else -3.0e30)
            x[5, width - 1] = 60000 if dtype is torch.float16 else 3.0e30
            yield f"{dtype}-width{width}", x


def check_host_topk_sample(op):
    """Reusable actual-op check; no backend import, device creation, or timing."""
    checked = 0
    for name, x in host_topk_sample_cases():
        before = x.clone()
        support = reference_probs(x) > 0
        torch.manual_seed(7)
        first = op(x, TEMPERATURE, TOP_K)
        torch.manual_seed(7)
        again = op(x, TEMPERATURE, TOP_K)
        assert first.device.type == "cpu" and first.dtype == torch.long, name
        assert tuple(first.shape) == (x.shape[0],), (name, tuple(first.shape))
        assert torch.equal(first, again), f"{name}: one seed drew two sequences"
        assert torch.equal(x.view(torch.uint8), before.view(torch.uint8)), name
        drawn = support.gather(1, first.reshape(-1, 1)).reshape(-1)
        assert bool(drawn.all()), (name, first[~drawn].tolist())
        checked += 1

    # Everything the tie rule keeps has to be reachable, and nothing else. A
    # flat row is the sharpest version: the whole row is one tie. Drawing the
    # identical rows in one call also fails a sampler that reuses a uniform.
    torch.manual_seed(11)
    columns = op(torch.full((4000, 64), 2.0), TEMPERATURE, TOP_K)
    assert set(columns.tolist()) == set(range(64)), sorted(set(range(64)) - set(columns.tolist()))

    # Mass: a two-level row has a closed-form ratio the draw must reproduce.
    torch.manual_seed(13)
    row = torch.zeros(40)
    row[:10] = 1.0
    draws = 30000
    expected = reference_probs(row.reshape(1, -1))[0]
    share = torch.bincount(op(row.expand(draws, 40).contiguous(), TEMPERATURE, TOP_K),
                           minlength=40).float() / draws
    assert float((share - expected).abs().max()) < 0.01, (share[:12], expected[:12])

    invalid = (
        ((torch.zeros((3, 4)).t(), TEMPERATURE, TOP_K), "contiguous"),
        ((torch.empty((1, 0)), TEMPERATURE, TOP_K), "empty last dimension"),
        ((torch.zeros(4), TEMPERATURE, TOP_K), "2-D"),
        ((torch.zeros((1, 2, 3)), TEMPERATURE, TOP_K), "2-D"),
        ((torch.zeros((1, 4), dtype=torch.float64), TEMPERATURE, TOP_K), "fp16 or fp32"),
        ((torch.zeros((1, 4), dtype=torch.bfloat16), TEMPERATURE, TOP_K), "fp16 or fp32"),
        ((torch.zeros((1, 4)), 0.0, TOP_K), "finite and positive"),
        ((torch.zeros((1, 4)), -1.0, TOP_K), "finite and positive"),
        ((torch.zeros((1, 4)), float("inf"), TOP_K), "finite and positive"),
        ((torch.zeros((1, 4)), TEMPERATURE, 0), "at least 1"),
        ((torch.tensor([[1.0, float("nan"), 3.0]]), TEMPERATURE, TOP_K), "non-finite"),
        ((torch.tensor([[1.0, float("inf")]]), TEMPERATURE, TOP_K), "non-finite"),
        ((torch.tensor([[1.0, -float("inf")]]), TEMPERATURE, TOP_K), "non-finite"),
    )
    for args, message in invalid:
        try:
            op(*args)
        except RuntimeError as error:
            assert message in str(error), (message, str(error))
        else:
            raise AssertionError(f"topk sample accepted an invalid {message!r} case")
    return {"valid_cases": checked, "rejected_cases": len(invalid)}


@pytest.fixture(scope="module")
def native_cpu_topk_sample(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    build = tmp_path_factory.mktemp("host-topk-sample")
    source = build / "sample.cpp"
    binary = build / "sample"
    source.write_text(r'''
#include "core/rpu_host_sample.h"
#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 9, "harness argument count");
    const std::string dtype = argv[1];
    const int rank = std::stoi(argv[2]);
    const int64_t rows = std::stoll(argv[3]), cols = std::stoll(argv[4]);
    const double temperature = std::stod(argv[5]);
    const int64_t top_k = std::stoll(argv[6]);
    at::manual_seed(std::stoull(argv[8]));
    at::Tensor input = at::empty({rows, cols}, at::kFloat);
    std::cin.read(reinterpret_cast<char*>(input.data_ptr<float>()), input.nbytes());
    if (dtype == "half") input = input.to(at::kHalf);
    if (dtype == "double") input = input.to(at::kDouble);
    if (dtype == "bfloat16") input = input.to(at::kBFloat16);
    if (rank == 1) input = input.reshape({cols});
    if (rank == 3) input = input.reshape({1, rows, cols});
    if (std::string(argv[7]) == "strided") {
      auto strided = at::empty({rows, cols * 2}, input.options()).slice(1, 0, cols * 2, 2);
      strided.copy_(input);
      input = strided;
    }
    auto before = input.clone();
    auto output = rpu::host_sample::sample_topk_lastdim_cpu(
        input, temperature, top_k, std::nullopt);
    TORCH_CHECK(output.device().is_cpu() && output.scalar_type() == at::kLong &&
                output.dim() == 1 && output.size(0) == rows && output.is_contiguous(),
                "native topk sample returned wrong metadata");
    TORCH_CHECK(input.nbytes() == 0 ||
                std::memcmp(input.data_ptr(), before.data_ptr(), input.nbytes()) == 0,
                "native topk sample changed its input bytes");
    std::cout.write(reinterpret_cast<const char*>(output.data_ptr<int64_t>()),
                    output.nbytes());
    return 0;
  } catch (const c10::Error& error) {
    std::cerr << error.what_without_backtrace();
    return 2;
  }
}
''')
    torch_root = Path(torch.__file__).resolve().parent
    command = [compiler, "-std=c++17", "-O2", "-pthread",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               "-I", str(ROOT / "src"), "-I", str(torch_root / "include"),
               str(source), "-L", str(torch_root / "lib"),
               f"-Wl,-rpath,{torch_root / 'lib'}", "-ltorch_cpu", "-lc10", "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr

    names = {torch.float16: "half", torch.float32: "float", torch.float64: "double",
             torch.bfloat16: "bfloat16"}

    def run(x, temperature, top_k):
        cols = x.shape[-1] if x.ndim else 1
        rows = x.numel() // cols if cols else 1
        # A subprocess cannot share this process's default generator, so the
        # seed the caller just set is forwarded and applied there instead.
        result = subprocess.run(
            [str(binary), names[x.dtype], str(x.ndim), str(rows), str(cols),
             repr(float(temperature)), str(int(top_k)),
             "contiguous" if x.is_contiguous() else "strided",
             str(int(torch.initial_seed()))],
            input=x.float().contiguous().numpy().tobytes(),
            capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.decode())
        return (torch.frombuffer(bytearray(result.stdout), dtype=torch.long).clone()
                if result.stdout else torch.empty(0, dtype=torch.long))
    return run


def test_native_topk_sample_matches_the_reference_distribution(native_cpu_topk_sample):
    assert check_host_topk_sample(native_cpu_topk_sample)["valid_cases"] == 12
