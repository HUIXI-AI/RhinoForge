from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import text as qwen3_5_text
from rpu_backend.adapters import qwen3
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError


ROOT = Path(__file__).resolve().parents[1]
EAGER_BMM_TILES = {
    (64, 64, 64),
    (64, 64, 128),
    (64, 128, 64),
    (64, 128, 128),
    (128, 128, 64),
    (128, 128, 128),
}


def test_eager_bmm_allowlist_matches_preloaded_kernels() -> None:
    bmm = (ROOT / "src/ops/rpu_bmm.cpp").read_text(encoding="utf-8")
    cache = (ROOT / "src/core/rpu_kernel_cache.inc").read_text(
        encoding="utf-8"
    )

    table = bmm.split("kPreloadedEagerBmmTiles[][3] = {", 1)[1].split(
        "};", 1
    )[0]
    assert {
        tuple(map(int, match))
        for match in re.findall(r"\{(\d+), (\d+), (\d+)\}", table)
    } == EAGER_BMM_TILES

    names = {
        (int(m), int(n), int(k), mode)
        for m, n, k, mode in re.findall(
            r'\{"gemm_fp16_spm_16b_w(\d+)x(\d+)_k(\d+)_1core_buf1_'
            r'nt_lpaddr_(peak|univ)"',
            cache,
        )
    }
    assert names == {
        (*tile, mode)
        for tile in EAGER_BMM_TILES
        for mode in ("peak", "univ")
    }

    eager = bmm.split("void rpu_launch_bmm_kernel", 1)[1].split(
        "// ============ BMM SPM Kernel Launch", 1
    )[0]
    assert eager.index("is_preloaded_eager_bmm_tile(") < eager.index(
        "std::string kernel_mode"
    )


@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
def test_qwen3_5_gdn_head_dims_fail_before_mutation(dims) -> None:
    model = SimpleNamespace(
        norm=SimpleNamespace(weight=torch.ones(1, dtype=torch.float16))
    )
    config = SimpleNamespace(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_key_head_dim=dims[0],
        linear_value_head_dim=dims[1],
    )

    with pytest.raises(UnsupportedModelError, match="GDN requires"):
        qwen3_5_text._install_qwen3_5_text_for_rpu_impl(
            model,
            config,
            max_seq_len=128,
            chunk_size_cap=0,
            exact_chunk_size=0,
            padding_budget=0,
            padding_rows="auto",
            execution_config={},
            graph_cache=object(),
            prefill_graph=object(),
            cpu_stage_weights=False,
        )

    assert not hasattr(model, "_rpu_qwen3_5_text_install_started")


def test_qwen3_5_gdn_guard_precedes_irreversible_marker() -> None:
    source = Path(qwen3_5_text.__file__).read_text(encoding="utf-8")
    install = source.split("def _install_qwen3_5_text_for_rpu_impl", 1)[1]
    assert install.index("if not all(is_full):") < install.index(
        "inner._rpu_qwen3_5_text_install_started = True"
    )


def test_qwen3_14b_requires_exact_lm_head_quantization_metadata() -> None:
    profile = {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 40,
        "num_key_value_heads": 8,
    }
    incomplete = SimpleNamespace(
        **profile,
        quant_config={"method": "w8a16"},
    )
    with pytest.raises(UnsupportedModelError, match="untied INT8 lm_head"):
        qwen3._check_profile(incomplete)

    exact = SimpleNamespace(
        **profile,
        quant_config={
            "method": "w8a16",
            "quantized_lm_head": True,
            "lm_head_untied": True,
            "quantized_embed_tokens": False,
        },
    )
    qwen3._check_profile(exact)

    model = torch.nn.Module()
    model.config = exact
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    model.model.embed_tokens = torch.nn.Embedding(2, 2).half()
    model.lm_head = torch.nn.Linear(2, 2).half()
    with pytest.raises(RPUBackendError, match="checkpoint tensors do not match"):
        qwen3.Qwen3Adapter(model)


def test_dinov3_example_preflights_before_weight_loading() -> None:
    source = (ROOT / "examples/dinov3.py").read_text(encoding="utf-8")
    assert source.index("DINOv3Adapter.preflight(model_hf_config)") < source.index(
        "DINOv3ViTModel.from_pretrained("
    )
