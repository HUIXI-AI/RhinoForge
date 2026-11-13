"""Isolated llama_gather_embedding kernel unit test (graph-coverage campaign,
embed-composition fold).

Strategy: build a small fp16 vocab table `[V, H]` + int32 ids `[N]`, run the device
kernel via `torch.ops.rpu.gather_embedding`, and compare against the CPU reference
`vocab.half()[ids]` (a pure row-gather — no scale; Qwen embedding_scale == 1.0).

The cos-sim is the ORACLE for the register/operand mapping (param packing taken
from rhino-runtime schemas/src/kernel_wrapper/r1/llama_common.cc:2635). A wrong
address split / param index shows up as a low cos here, isolated from the VLA.

Cases:
  - random ids over the full vocab → exercises arbitrary row gather.
  - id 0 and id V-1 present → boundary rows.
  - bit-exact expectation: the gather is a pure copy of fp16 rows, so cos should
    be 1.0 (not just >= 0.999) once the addressing is correct.

Run:
    sudo LD_PRELOAD=$(g++ -print-file-name=libstdc++.so.6) \
        python \
        python/rpu_backend/tests/ops/test_embedding_gather_isolated.py
"""
from __future__ import annotations

import sys

import pytest
import torch
import torch.nn.functional as F

import rpu_backend  # noqa: F401

SEED = 20260614
V = 320          # small vocab (mult of 16 not required for V)
H = 2048         # wall_oss hidden_size (mult of 16)
N = 180          # wall_oss single-image prefix length


def _run_case(name: str, ids_cpu: torch.Tensor, vocab_cpu: torch.Tensor) -> bool:
    # CPU reference: gather fp16 rows (device stores the table in fp16).
    ref = vocab_cpu.half()[ids_cpu.long()].float()              # [N, H]

    vocab_rpu = vocab_cpu.half().to("rpu")
    ids_rpu = ids_cpu.to(torch.int32).to("rpu")
    out = torch.ops.rpu.gather_embedding(vocab_rpu, ids_rpu)     # [N, H] fp16 on device
    assert out.dtype == torch.float16
    out_cpu = out.float().cpu()

    assert out_cpu.shape == ref.shape, f"{name}: shape {tuple(out_cpu.shape)} != {tuple(ref.shape)}"
    cos = F.cosine_similarity(out_cpu.reshape(-1), ref.reshape(-1), dim=0).item()
    mse = (out_cpu - ref).pow(2).mean().item()
    max_abs = (out_cpu - ref).abs().max().item()
    ok = cos >= 0.999 and torch.equal(out_cpu, ref) and torch.isfinite(out_cpu).all().item()
    print(f"  [{name}] cos={cos:.7f} mse={mse:.3e} max_abs={max_abs:.3e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def _run_assemble(name: str, ids_cpu, vocab_cpu, n_img: int, img_start: int) -> bool:
    # CPU reference: gather fp16 rows, overwrite [img_start:img_start+n_img] with merged.
    ref = vocab_cpu.half()[ids_cpu.long()].float()                  # [N, H]
    merged_cpu = torch.randn(n_img, H, dtype=torch.float32)
    ref[img_start:img_start + n_img] = merged_cpu.half().float()

    vocab_rpu = vocab_cpu.half().to("rpu")
    ids_rpu = ids_cpu.to(torch.int32).to("rpu")
    merged_rpu = merged_cpu.half().to("rpu")
    out = torch.ops.rpu.assemble_inputs_embeds(vocab_rpu, ids_rpu, merged_rpu, img_start)
    assert out.dtype == torch.float16
    out_cpu = out.float().cpu()
    assert out_cpu.shape == ref.shape, f"{name}: shape {tuple(out_cpu.shape)} != {tuple(ref.shape)}"
    mse = (out_cpu - ref).pow(2).mean().item()
    max_abs = (out_cpu - ref).abs().max().item()
    ok = max_abs < 1e-3 and torch.equal(out_cpu, ref) and torch.isfinite(out_cpu).all().item()
    print(f"  [{name}] mse={mse:.3e} max_abs={max_abs:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    torch.manual_seed(SEED)
    vocab = torch.randn(V, H, dtype=torch.float32)

    ok = True
    # Case 1: random ids
    ids1 = torch.randint(0, V, (N,), dtype=torch.int64)
    ok &= _run_case("random", ids1, vocab)

    # Case 2: boundary ids (0 and V-1 present), shorter
    ids2 = torch.tensor([0, V - 1, 1, V - 2] + list(range(0, 28)), dtype=torch.int64)
    ok &= _run_case("boundary", ids2, vocab)

    # Case 3: fused assemble — gather + scatter a contiguous image block (wall_oss shape:
    # 64 vision tokens at offset 5 inside a 192-padded prefix).
    ids3 = torch.randint(0, V, (192,), dtype=torch.int64)
    ok &= _run_assemble("assemble", ids3, vocab, n_img=64, img_start=5)

    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


# The pytest entry reuses the script checks and shares import/kernel-cache setup.
# Axis: llama_gather_embedding (wall-oss patch_embed on-device gather).
def test_embedding_gather(record_property):
    assert main() == 0


def run_embedding_parts_regression():
    """Exact device composition with shared-owner slices and fail-before-launch inputs."""
    generator = torch.Generator().manual_seed(20261003)
    vocab = torch.randn(320, 2048, generator=generator).half()
    ids = torch.randint(0, 320, (480,), generator=generator, dtype=torch.int32)
    source = torch.randn(440, 2048, generator=generator).half()
    vocab_r, ids_r, source_r = vocab.to("rpu"), ids.to("rpu"), source.to("rpu")
    offsets = (3, 149, 294)
    runs = torch.tensor([[5, 144], [160, 144], [320, 144]], dtype=torch.int64)
    parts = [source_r[start:start + 144] for start in offsets]
    outputs, references = [], []
    for changed in (False, True, False):
        current = source + 1 if changed else source
        source_r.copy_(current)
        order = (2, 0, 1) if changed else (0, 1, 2)
        call_runs = runs[list(order)].contiguous()
        call_parts = [parts[index] for index in order]
        expected = vocab[ids.long()].clone()
        for index in order:
            start, length = runs[index].tolist()
            expected[start:start + length] = current[offsets[index]:offsets[index] + length]
        result = torch.ops.rpu.assemble_inputs_embeds_parts(vocab_r, ids_r, call_parts, call_runs)
        assert result.shape == expected.shape and result.dtype == torch.float16
        assert result.device.type == "rpu"
        outputs.append(result)
        references.append(expected)
    assert len({result.data_ptr() for result in outputs}) == 3
    for result, expected in zip(outputs, references):
        assert torch.equal(result.cpu(), expected)
    assert not torch.equal(outputs[0].cpu(), outputs[1].cpu())
    assert torch.equal(outputs[0].cpu(), outputs[2].cpu())

    # A preassembled token skeleton uses DMA rather than a redundant gather.
    # Its ids carry only the output length; image parts still own their rows.
    skeleton = vocab[ids.long()].clone()
    skeleton_r = skeleton.to("rpu")
    unused_ids = torch.full_like(ids, -1).to("rpu")
    copied, copy_references = [], []
    for changed in (False, True, False):
        current = source + 1 if changed else source
        source_r.copy_(current)
        skeleton_r.copy_(skeleton + (0.5 if changed else 0))
        expected = skeleton + (0.5 if changed else 0)
        for index, (start, length) in enumerate(runs.tolist()):
            expected[start:start + length] = current[offsets[index]:offsets[index] + length]
        copied.append(torch.ops.rpu.assemble_inputs_embeds_parts(
            skeleton_r, unused_ids, parts, runs, True))
        copy_references.append(expected)
    assert len({result.data_ptr() for result in copied}) == 3
    for result, expected in zip(copied, copy_references):
        assert torch.equal(result.cpu(), expected)
    with pytest.raises(RuntimeError, match="prefix rows"):
        torch.ops.rpu.assemble_inputs_embeds_parts(vocab_r, ids_r, parts, runs, True)
    restored_prefix = torch.ops.rpu.assemble_inputs_embeds_parts(
        skeleton_r, unused_ids, parts, runs, True)
    assert torch.equal(restored_prefix.cpu(), copy_references[0])
    assert torch.equal(copied[0].cpu(), copy_references[0])

    # Every malformed list is fully rejected before the first gather or DMA.
    bad_runs = (
        torch.tensor([[-1, 144], [160, 144], [320, 144]]),
        torch.tensor([[5, 0], [160, 144], [320, 144]]),
        torch.tensor([[5, 144], [160, 144], [400, 144]]),
        torch.tensor([[5, 144], [100, 144], [320, 144]]),
        torch.tensor([[5, 144], [160, 144], [320, 2**63 - 1]]),
        runs.to(torch.int32), runs.to("rpu"), runs[:, :1],
        torch.stack((runs, runs), dim=-1)[..., 0],
    )
    rejected = 0
    for invalid in bad_runs:
        with pytest.raises(RuntimeError, match="assemble_inputs_embeds_parts"):
            torch.ops.rpu.assemble_inputs_embeds_parts(vocab_r, ids_r, parts, invalid)
        rejected += 1
    invalid_parts = (
        parts[:-1], [parts[0], parts[1], parts[2][:-1]],
        [parts[0], parts[1], source[:144]],
        [parts[0], parts[1], parts[2].to(torch.float32)],
        [parts[0], parts[1], source_r[:144, ::2]],
        [parts[0], parts[1], torch.empty(144 * 2048 + 1, dtype=torch.float16,
                                       device="rpu")[1:].view(144, 2048)],
    )
    for invalid in invalid_parts:
        with pytest.raises(RuntimeError, match="assemble_inputs_embeds_parts"):
            torch.ops.rpu.assemble_inputs_embeds_parts(vocab_r, ids_r, invalid, runs)
        rejected += 1
    for invalid_vocab, invalid_ids in (
        (vocab, ids_r), (vocab_r, ids), (vocab_r, ids_r.to(torch.int64)),
        (vocab_r, ids_r.view(1, -1)), (vocab_r.to(torch.float32), ids_r),
        (vocab_r, torch.zeros(481, dtype=torch.int32).to("rpu")[1:]),
        (torch.empty(320 * 2048 + 1, dtype=torch.float16, device="rpu")[1:].view(320, 2048), ids_r),
    ):
        with pytest.raises(RuntimeError, match="assemble_inputs_embeds_parts"):
            torch.ops.rpu.assemble_inputs_embeds_parts(invalid_vocab, invalid_ids, parts, runs)
        rejected += 1
    # A final valid call demonstrates the rejected requests did not poison SPM.
    restored = torch.ops.rpu.assemble_inputs_embeds_parts(vocab_r, ids_r, parts, runs)
    assert torch.equal(restored.cpu(), references[0])
    assert torch.equal(outputs[0].cpu(), references[0])
    return {"same_input_exact": True, "changed_input_observed": True,
            "fresh_outputs_retained": True, "rejected_inputs": rejected}


def test_embedding_parts(record_property):
    run_embedding_parts_regression()
