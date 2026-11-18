"""Wiring for the fused top-k draw that the CPU-native semantic test cannot see.

`test_host_topk_sample` compiles the shipped arithmetic and checks its
distribution, but it calls the header entry directly. The dispatcher boundary,
the RPU hand-over, the Graph RNG classification and the fake-tensor shape rule
only exist in these sources, and a sampling op that is misclassified for Graph
capture replays one frozen token forever.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = (ROOT / "src/core/rpu_backend.cpp").read_text(encoding="utf-8")
DISPATCH = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text(encoding="utf-8")
CMAKE = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
DECLS = (ROOT / "src/core/rpu_kernel_decls.h").read_text(encoding="utf-8")
HOST = (ROOT / "src/core/rpu_host_sample.h").read_text(encoding="utf-8")
GRAPH = (ROOT / "src/graph/graph_infra.cpp").read_text(encoding="utf-8")
FAKE = (ROOT / "python/rpu_backend/graph/fake_dispatch.py").read_text(encoding="utf-8")
TTS = (ROOT / "python/rpu_backend/api/qwen3_tts.py").read_text(encoding="utf-8")


def dispatch_block(name):
    return DISPATCH.split(f"TORCH_LIBRARY_IMPL(rpu, {name}, m) {{", 1)[1].split("\n}", 1)[0]


def test_fused_draw_is_registered_on_both_devices_without_a_new_translation_unit():
    assert '#include "rpu_host_sample.h"' in BACKEND
    assert "rpu::host_sample::sample_topk_lastdim_cpu(" in BACKEND
    assert "rpu_host_sample.cpp" not in CMAKE  # header-only, as the argmax scanner is
    assert "at::Tensor rpu_sample_topk_lastdim_host(const at::Tensor &self, double temperature," in DECLS
    schema = ('m.def("sample_topk_lastdim_host(Tensor self, float temperature, int top_k, "\n'
              '        "Generator? generator=None) -> Tensor");')
    assert schema in DISPATCH
    impl = 'm.impl("sample_topk_lastdim_host", TORCH_FN(rpu_sample_topk_lastdim_host));'
    assert impl in dispatch_block("PrivateUse1")
    assert impl in dispatch_block("CPU")
    assert ('m.impl("sample_topk_lastdim_host", torch::CppFunction::makeFallthrough());'
            in dispatch_block("AutogradPrivateUse1"))


def test_rpu_logits_stay_fp16_and_are_read_through_the_flushed_host_alias():
    entry = BACKEND.split("at::Tensor rpu_sample_topk_lastdim_host(", 1)[1].split("\n}\n", 1)[0]
    assert "host topk sample: logits must be on CPU or RPU" in entry
    assert "host topk sample: RPU logits must be fp16" in entry
    assert "rpu_to_cpu_zerocopy(logits, /*flush=*/true)" in entry
    # The CPU entry is the one that validates dtype/shape for the CPU case, so
    # the RPU branch must not be the only place those checks live.
    cpu_entry = HOST.split("inline at::Tensor sample_topk_lastdim_cpu(", 1)[1]
    assert 'TORCH_CHECK(logits.device().is_cpu(),' in cpu_entry
    assert "logits must be fp16 or fp32" in cpu_entry


def test_the_draw_is_classified_as_rng_and_shaped_for_fake_tensors():
    rng = GRAPH.split("rng_op_set()", 1)[1].split("};", 1)[0]
    assert '"rpu::sample_topk_lastdim_host",' in rng
    meta = FAKE.split('@torch.library.impl("rpu::sample_topk_lastdim_host", "Python")', 1)[1]
    meta = meta.split("@torch.library.impl", 1)[0]
    assert "def _meta_sample_topk_lastdim_host(logits, temperature, top_k, generator=None):" in meta
    assert "_require_fake(logits)" in meta
    assert "tuple(logits.shape[:-1])" in meta and "dtype=torch.long" in meta
    assert 'device="cpu"' in meta


def test_the_predictor_route_draws_through_the_op_and_keeps_no_warper_chain():
    route = TTS.split("def _predictor_generate(", 1)[1].split("\ndef ", 1)[0]
    assert "torch.ops.rpu.sample_topk_lastdim_host(" in route
    assert "LogitsWarper" not in route and "multinomial" not in route
    assert "temperature, top_k" in route  # the checkpoint's own values, not constants
