"""Host-only allow-list for Graph-invisible Queue/SDK DMA escape hatches."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CPP_SUFFIXES = {".cpp", ".cc", ".h", ".hpp", ".inc"}

BROADCAST = "rpu_launch_ddr_broadcast_spm_dma_immediate"
SPM_COPY = "rpu_launch_spm_copy_ddr_dma_immediate"
BYTE_COPY = "rpu_launch_ddr_spm_bytes_immediate"
DDR_SCATTER = "rpu_launch_ddr_scatter_spm_dma_immediate"
SPM_SCATTER = "rpu_launch_spm_scatter_ddr_dma_immediate"
IMMEDIATE_APIS = (
    BROADCAST,
    SPM_COPY,
    BYTE_COPY,
    DDR_SCATTER,
    SPM_SCATTER,
)

RAW_QUEUE_TYPE = r"(?:(?:::)?(?:[A-Za-z_]\w*::)*)Queue_t"
RAW_QUEUE_CACHE_TYPE = r"(?:(?:::)?(?:[A-Za-z_]\w*::)*)QueueCache"

QUEUE_CACHE_GET_OWNERS = (
    ("src/graph/graph_runtime_execute.cpp", "RpuKernelGraph::execute_graph_oneshot"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_broadcast_spm_dma_mutable"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_scatter_spm_dma_mutable"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_scatter_ddr_dma_mutable"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_copy_ddr_dma_mutable"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_scatter_spm_dma"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_local_shard_copy_dma"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_gather_spm_dma"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_broadcast_spm_dma_immediate"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_copy_ddr_dma_immediate"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_spm_bytes_immediate"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_scatter_spm_dma_immediate"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_scatter_ddr_dma_immediate"),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_assembled_prefix_parts_dma_immediate"),
)

BATCH_LIFECYCLE_OWNERS: dict[tuple[str, str], Counter[str]] = {
    (
        "src/graph/graph_runtime_execute.cpp",
        "RpuKernelGraph::execute_ddr_copy",
    ): Counter(build_batch=1, enqueu_batch=1, discard_completed_batch=1),
    (
        "src/graph/graph_runtime_execute.cpp",
        "RpuKernelGraph::prepare_segment_queue",
    ): Counter(build_batch=1),
    (
        "src/graph/graph_runtime_execute.cpp",
        "RpuKernelGraph::launch_segment_for_replay",
    ): Counter(enqueu_batch=1),
    (
        "src/graph/graph_runtime_execute.cpp",
        "RpuKernelGraph::launch_segment_sync_only",
    ): Counter(enqueu_batch=1),
    (
        "src/graph/graph_runtime_execute.cpp",
        "RpuKernelGraph::execute_graph_for_recording",
    ): Counter(enqueu_batch=1),
    (
        "src/graph/graph_runtime_execute.cpp",
        "RpuKernelGraph::execute_graph_oneshot",
    ): Counter(enqueu_batch=1, discard_completed_batch=1),
    (
        "src/ops/rpu_memcpy.cpp",
        "submit_immediate_batch",
    ): Counter(build_batch=1, enqueu_batch=1, discard_completed_batch=1),
}

ADD_BARRIER_OWNERS: dict[tuple[str, str], int] = {
    (
        "src/graph/graph_runtime_execute.cpp",
        "RpuKernelGraph::prepare_segment_queue",
    ): 1,
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_broadcast_spm_dma_mutable"): 2,
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_scatter_spm_dma_mutable"): 2,
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_scatter_ddr_dma_mutable"): 2,
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_scatter_spm_dma"): 2,
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_local_shard_copy_dma"): 2,
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_gather_spm_dma"): 2,
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_broadcast_spm_dma_immediate"): 2,
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_scatter_spm_dma_immediate"): 2,
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_scatter_ddr_dma_immediate"): 2,
}

RAW_QUEUE_SDK_TOKEN_INVENTORY: dict[str, Counter[str]] = {
    "build_batch": Counter(
        {
            "src/graph/graph_runtime_execute.cpp": 2,
            "src/ops/rpu_memcpy.cpp": 1,
        }
    ),
    "enqueu_batch": Counter(
        {
            "src/graph/graph_runtime_execute.cpp": 5,
            "src/ops/rpu_memcpy.cpp": 1,
        }
    ),
    "discard_completed_batch": Counter(
        {
            "src/graph/graph_runtime_execute.cpp": 2,
            "src/ops/rpu_memcpy.cpp": 1,
        }
    ),
    "add_dma_kernel": Counter(
        {
            "src/core/rpu_kernel_cache.h": 1,
        }
    ),
    "add_dma_kernel_mutable": Counter(
        {"src/core/rpu_kernel_cache.h": 1}
    ),
    "update_dma_kernel": Counter(
        {"src/graph/graph_runtime_execute.cpp": 1}
    ),
    "add_barrier": Counter(
        {
            "src/graph/graph_runtime_execute.cpp": 1,
            "src/ops/rpu_memcpy.cpp": 18,
        }
    ),
}


def _blank_preserving_newlines(value: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in value)


_RAW_STRING = re.compile(
    r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"',
    re.DOTALL,
)


def _strip_cpp_comments_and_literals(source: str) -> str:
    """Blank comments and common C++ literals while preserving line offsets."""

    source = _RAW_STRING.sub(
        lambda match: _blank_preserving_newlines(match.group(0)), source
    )
    out = list(source)
    index = 0
    state = "code"
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char == "/" and nxt == "/":
                out[index] = out[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if char == "/" and nxt == "*":
                out[index] = out[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if char == '"':
                out[index] = " "
                index += 1
                state = "string"
                continue
            if char == "'":
                out[index] = " "
                index += 1
                state = "char"
                continue
            index += 1
            continue
        if state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                out[index] = " "
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and nxt == "/":
                out[index] = out[index + 1] = " "
                index += 2
                state = "code"
            else:
                if char != "\n":
                    out[index] = " "
                index += 1
            continue
        if state in {"string", "char"}:
            terminator = '"' if state == "string" else "'"
            if char == "\\":
                out[index] = " "
                if index + 1 < len(source):
                    if source[index + 1] != "\n":
                        out[index + 1] = " "
                    index += 2
                else:
                    index += 1
            elif char == terminator:
                out[index] = " "
                index += 1
                state = "code"
            else:
                if char != "\n":
                    out[index] = " "
                index += 1
            continue
    return "".join(out)


def _sources() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): _strip_cpp_comments_and_literals(
            path.read_text(encoding="utf-8")
        )
        for path in SRC.rglob("*")
        if path.is_file() and path.suffix in CPP_SUFFIXES
    }


def _match_close(source: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(source)):
        if source[index] == opening:
            depth += 1
        elif source[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError(f"unclosed {opening!r} at offset {start}")


def _function_span(source: str, name: str) -> tuple[int, int, int]:
    """Return (name, body_begin, body_end) for one ordinary C++ definition."""

    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(")
    for match in pattern.finditer(source):
        open_paren = source.find("(", match.start())
        close_paren = _match_close(source, open_paren, "(", ")")
        cursor = close_paren + 1
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        brace = source.find("{", cursor)
        semicolon = source.find(";", cursor)
        if brace < 0 or (semicolon >= 0 and semicolon < brace):
            continue
        return match.start(), brace + 1, _match_close(source, brace, "{", "}")
    raise AssertionError(f"definition not found: {name}")


def _is_function_declaration(source: str, match_start: int) -> bool:
    """Return whether a function-like token is followed by a plain `;`."""

    open_paren = source.find("(", match_start)
    close_paren = _match_close(source, open_paren, "(", ")")
    cursor = close_paren + 1
    while cursor < len(source) and source[cursor].isspace():
        cursor += 1
    return cursor < len(source) and source[cursor] == ";"


def _count_by_file(sources: dict[str, str], pattern: str) -> Counter[str]:
    regex = re.compile(pattern, re.MULTILINE)
    return Counter(
        {
            path: len(tuple(regex.finditer(source)))
            for path, source in sources.items()
            if regex.search(source)
        }
    )


def _c(**counts: int) -> Counter[str]:
    aliases = {
        "broadcast": BROADCAST,
        "spm_copy": SPM_COPY,
        "byte_copy": BYTE_COPY,
        "ddr_scatter": DDR_SCATTER,
        "spm_scatter": SPM_SCATTER,
    }
    return Counter({aliases[name]: value for name, value in counts.items()})


# Exact source-level inventory of public immediate DMA call sites.
IMMEDIATE_CALLERS: dict[tuple[str, str], Counter[str]] = {
    ("src/ops/rpu_all_gather.cpp", "rpu_all_gather_test"): _c(
        ddr_scatter=1, spm_copy=1
    ),
    ("src/ops/rpu_bmm.cpp", "rpu_bmm_spm_test"): _c(
        broadcast=2, spm_copy=1
    ),
    ("src/ops/rpu_cast.cpp", "rpu_cast_uint8_fp16"): _c(
        byte_copy=1, spm_copy=1
    ),
    ("src/ops/rpu_cast.cpp", "rpu_cast_uint8_fp16_into"): _c(
        byte_copy=1, spm_copy=1
    ),
    ("src/ops/rpu_cumsum.cpp", "rpu_cumsum_test"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_eltwise_binary.cpp", "rpu_eltwise_bx1xc_test"): _c(
        broadcast=2, spm_copy=1
    ),
    ("src/ops/rpu_eltwise_unary.cpp", "rpu_gelu_erf"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_eltwise_unary.cpp", "rpu_launch_eltwise_unary_kernel"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_embedding.cpp", "rpu_gather_embedding"): _c(spm_copy=1),
    ("src/ops/rpu_embedding.cpp", "rpu_assemble_inputs_embeds"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_embedding.cpp", "rpu_assemble_inputs_embeds_parts"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_embedding.cpp", "rpu_assemble_inputs_embeds_rev"): _c(
        spm_copy=1
    ),
    ("src/ops/rpu_fast_pos_embed.cpp", "rpu_fast_pos_embed_interp_test"): _c(
        spm_copy=1
    ),
    ("src/ops/rpu_fla_conv1d.cpp", "rpu_fla_conv1d_test"): _c(
        broadcast=2, spm_copy=1
    ),
    ("src/ops/rpu_fla_conv1d.cpp", "rpu_fla_conv1d_prefill_test"): _c(
        broadcast=2, spm_copy=2
    ),
    ("src/ops/rpu_gdn_recurrent.cpp", "rpu_gdn_recurrent_multihead_test"): _c(
        broadcast=1, spm_copy=2
    ),
    ("src/ops/rpu_grouped_parallel_linear.cpp", "rpu_grouped_parallel_linear_test"): _c(
        broadcast=1, ddr_scatter=2, spm_scatter=1
    ),
    ("src/ops/rpu_hyvla_merger.cpp", "stage_members"): _c(ddr_scatter=1),
    ("src/ops/rpu_hyvla_merger.cpp", "rpu_hyvla_merger_pool"): _c(
        spm_scatter=1
    ),
    ("src/ops/rpu_hyvla_merger.cpp", "rpu_hyvla_merger_combine"): _c(
        spm_scatter=1
    ),
    ("src/ops/rpu_l2norm.cpp", "rpu_l2norm_test"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_linear.cpp", "tts_codec_im2col_rows"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_linear.cpp", "tts_codec_layernorm_rows"): _c(
        broadcast=3, spm_copy=1
    ),
    ("src/ops/rpu_partial_mrope.cpp", "rpu_partial_mrope_test"): _c(
        broadcast=1, spm_copy=1, ddr_scatter=1, spm_scatter=1
    ),
    ("src/ops/rpu_partial_mrope.cpp", "rpu_partial_rope_1d_test"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_permute.cpp", "rpu_permute3d_spm_test"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_permute.cpp", "rpu_permute4d_0213_spm_test"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_rope_2d.cpp", "rpu_rope_2d_test_ddr"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_rope_2d.cpp", "rpu_rope_2d_test_spm"): _c(
        broadcast=3, spm_copy=1
    ),
    ("src/ops/rpu_rmsnorm.cpp", "rpu_rmsnorm_spm_test"): _c(
        broadcast=2, spm_copy=1
    ),
    ("src/ops/rpu_sdpa.cpp", "sdpa_dma_mask_to_spm"): _c(broadcast=1),
    ("src/ops/rpu_sdpa.cpp", "rpu_temporal_sdpa_composite_test"): _c(
        ddr_scatter=3, spm_scatter=1
    ),
    ("src/ops/rpu_slice.cpp", "rpu_slice_spm_test"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_tile.cpp", "rpu_tile_spm_test"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_unit_tril_inv.cpp", "rpu_unit_tril_inv_test"): _c(
        broadcast=1, spm_copy=1
    ),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_broadcast_spm_dma"): _c(
        broadcast=1
    ),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_ddr_scatter_spm_dma"): _c(
        ddr_scatter=1
    ),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_scatter_ddr_dma"): _c(
        spm_scatter=1
    ),
    ("src/ops/rpu_memcpy.cpp", "rpu_launch_spm_copy_ddr_dma"): _c(
        spm_copy=1
    ),
}


def test_raw_queue_and_sdk_dma_surface_is_allowlisted() -> None:
    sources = _sources()

    assert _count_by_file(
        sources, rf"\bnew\s+{RAW_QUEUE_TYPE}\s*[\(\{{]"
    ) == Counter({"src/core/rpu_kernel_cache.h": 1})
    assert _count_by_file(
        sources,
        rf"(?<![A-Za-z0-9_])"
        rf"(?:(?:::)?(?:[A-Za-z_]\w*::)*)make_unique\s*<\s*"
        rf"{RAW_QUEUE_TYPE}\s*>\s*[\(\{{]",
    ) == Counter({"src/graph/graph_runtime_execute.cpp": 3})
    # Freeze direct temporary/heap type-initializers and named stack objects
    # separately.  The direct form intentionally sees the one allowed `new`
    # construction; any additional `Queue_t(...)` / `Queue_t{...}` temporary
    # or heap construction changes this exact inventory.
    assert _count_by_file(
        sources,
        rf"(?<![A-Za-z0-9_]){RAW_QUEUE_TYPE}\s*[\(\{{]",
    ) == Counter({"src/core/rpu_kernel_cache.h": 1})
    assert not _count_by_file(
        sources,
        rf"(?<![A-Za-z0-9_]){RAW_QUEUE_TYPE}\s+[A-Za-z_]\w*\s*[\(\{{]",
    )
    # Alternate type names would make all constructor spellings above blind.
    # Same-name convenience aliases remain visible as `Queue_t`; only aliases
    # that rename the SDK type are forbidden.
    assert not _count_by_file(
        sources,
        rf"\busing\s+(?!Queue_t\b)[A-Za-z_]\w*\s*=\s*"
        rf"{RAW_QUEUE_TYPE}\s*;",
    )
    assert not _count_by_file(
        sources,
        rf"\btypedef\s+{RAW_QUEUE_TYPE}\s+"
        rf"(?!Queue_t\b)[A-Za-z_]\w*\s*;",
    )
    assert not _count_by_file(
        sources,
        rf"^\s*#\s*define\s+(?!Queue_t\b)[A-Za-z_]\w*\s+"
        rf"{RAW_QUEUE_TYPE}\b",
    )
    assert not _count_by_file(
        sources,
        rf"(?<![A-Za-z0-9_])"
        rf"(?:(?:::)?(?:[A-Za-z_]\w*::)*)"
        rf"(?:make_shared|allocate_shared|construct_at|"
        rf"make_unique_for_overwrite)\s*<\s*{RAW_QUEUE_TYPE}\s*>\s*[\(\{{]",
    )
    assert not _count_by_file(
        sources,
        rf"\busing\s+(?!QueueCache\b)[A-Za-z_]\w*\s*=\s*"
        rf"{RAW_QUEUE_CACHE_TYPE}\s*;",
    )
    assert not _count_by_file(
        sources,
        rf"\btypedef\s+{RAW_QUEUE_CACHE_TYPE}\s+"
        rf"(?!QueueCache\b)[A-Za-z_]\w*\s*;",
    )
    assert not _count_by_file(
        sources,
        rf"^\s*#\s*define\s+(?!QueueCache\b)[A-Za-z_]\w*\s+"
        rf"{RAW_QUEUE_CACHE_TYPE}\b",
    )

    queue_cache_instance = (
        r"(?<![A-Za-z0-9_])QueueCache::instance\s*\(\s*\)"
    )
    expected_queue_cache_instances = Counter(
        {
            "src/core/rpu_backend.cpp": 1,
            "src/core/rpu_kernel_cache.h": 1,
            "src/ops/rpu_memcpy.cpp": 14,  # one new get plus failure-only clear
            "src/graph/graph_runtime_execute.cpp": 2,  # one-shot get and failure eviction
        }
    )
    # Freeze the singleton expression itself, not only `.get` / `.clear`, so
    # assigning `QueueCache::instance()` to an alias cannot evade this gate.
    assert _count_by_file(sources, queue_cache_instance) == (
        expected_queue_cache_instances
    )

    queue_cache_get = re.compile(
        queue_cache_instance + r"\s*\.get\s*\("
    )
    assert _count_by_file(sources, queue_cache_get.pattern) == Counter(
        {
            "src/core/rpu_kernel_cache.h": 1,
            "src/ops/rpu_memcpy.cpp": 13,
            "src/graph/graph_runtime_execute.cpp": 1,
        }
    )
    assert _count_by_file(
        sources, queue_cache_instance + r"\s*\.clear\s*\("
    ) == Counter({"src/core/rpu_backend.cpp": 1, "src/ops/rpu_memcpy.cpp": 1})

    queue_cache_evict = queue_cache_instance + r"\s*\.evict\s*\("
    assert _count_by_file(sources, queue_cache_evict) == Counter({
        "src/graph/graph_runtime_execute.cpp": 1,
    })
    _, begin, end = _function_span(
        sources["src/graph/graph_runtime_execute.cpp"],
        "RpuKernelGraph::execute_graph_oneshot")
    assert len(re.findall(queue_cache_evict,
                          sources["src/graph/graph_runtime_execute.cpp"][begin:end])) == 1

    queue_get_spans: dict[str, list[tuple[str, int, int]]] = {}
    for path, function in QUEUE_CACHE_GET_OWNERS:
        _, begin, end = _function_span(sources[path], function)
        queue_get_spans.setdefault(path, []).append((function, begin, end))
        assert len(tuple(queue_cache_get.finditer(sources[path][begin:end]))) == 1
    for path, source in sources.items():
        for match in queue_cache_get.finditer(source):
            if path == "src/core/rpu_kernel_cache.h":
                assert source.rfind("#define GET_QUEUE", 0, match.start()) >= 0
                continue
            owners = [
                function
                for function, begin, end in queue_get_spans.get(path, ())
                if begin <= match.start() < end
            ]
            assert len(owners) == 1, (path, match.start(), owners)

    raw_get_queue = _count_by_file(
        sources, r"^\s*#\s*define\s+GET_QUEUE\s*\("
    )
    assert raw_get_queue == Counter({
        "src/core/rpu_kernel_cache.h": 1,
        "src/graph/graph_runtime.h": 1,
    })
    assert "RpuQueueCache::instance().get" in sources["src/graph/graph_runtime.h"]

    assert _count_by_file(
        sources, r"(?:->|\.)\s*add_dma_kernel\s*\("
    ) == Counter(
        {
            "src/core/rpu_kernel_cache.h": 1,
        }
    )
    assert _count_by_file(
        sources, r"(?:->|\.)\s*add_dma_kernel_mutable\s*\("
    ) == Counter({"src/core/rpu_kernel_cache.h": 1})
    assert _count_by_file(
        sources, r"(?:->|\.)\s*update_dma_kernel\s*\("
    ) == Counter({"src/graph/graph_runtime_execute.cpp": 1})

    assert _count_by_file(
        sources, r"(?:->|\.)\s*build_batch\s*\("
    ) == Counter(
        {
            "src/graph/graph_runtime_execute.cpp": 2,
            "src/ops/rpu_memcpy.cpp": 1,
        }
    )
    assert _count_by_file(
        sources, r"(?:->|\.)\s*enqueu_batch\s*\("
    ) == Counter(
        {
            "src/graph/graph_runtime_execute.cpp": 5,
            "src/ops/rpu_memcpy.cpp": 1,
        }
    )
    assert _count_by_file(
        sources, r"(?:->|\.)\s*discard_completed_batch\s*\("
    ) == Counter(
        {
            "src/graph/graph_runtime_execute.cpp": 2,
            "src/ops/rpu_memcpy.cpp": 1,
        }
    )

    batch_lifecycle = re.compile(
        r"(?:->|\.)\s*"
        r"(?P<method>build_batch|enqueu_batch|discard_completed_batch)\s*\("
    )
    batch_spans: dict[str, list[tuple[str, int, int]]] = {}
    for (path, function), expected in BATCH_LIFECYCLE_OWNERS.items():
        _, begin, end = _function_span(sources[path], function)
        batch_spans.setdefault(path, []).append((function, begin, end))
        observed = Counter(
            match.group("method")
            for match in batch_lifecycle.finditer(sources[path][begin:end])
        )
        assert observed == expected, (path, function, observed, expected)
    for path, source in sources.items():
        for match in batch_lifecycle.finditer(source):
            owners = [
                function
                for function, begin, end in batch_spans.get(path, ())
                if begin <= match.start() < end
            ]
            assert len(owners) == 1, (
                path,
                match.start(),
                match.group("method"),
                owners,
            )

    add_barrier = re.compile(r"(?:->|\.)\s*add_barrier\s*\(")
    barrier_spans: dict[str, list[tuple[str, int, int]]] = {}
    for (path, function), expected in ADD_BARRIER_OWNERS.items():
        _, begin, end = _function_span(sources[path], function)
        barrier_spans.setdefault(path, []).append((function, begin, end))
        observed = len(tuple(add_barrier.finditer(sources[path][begin:end])))
        assert observed == expected, (path, function, observed, expected)
    for path, source in sources.items():
        for match in add_barrier.finditer(source):
            owners = [
                function
                for function, begin, end in barrier_spans.get(path, ())
                if begin <= match.start() < end
            ]
            assert len(owners) == 1, (path, match.start(), owners)

    # Bare-token counts close function-pointer and macro indirection holes: a
    # direct call removed in favor of an alias fails the call/owner inventory,
    # while retaining it and adding an alias fails this exact token inventory.
    for method, expected in RAW_QUEUE_SDK_TOKEN_INVENTORY.items():
        assert _count_by_file(sources, rf"\b{method}\b") == expected, method

    assert _count_by_file(sources, r"\bCopyToDevice\s*\(") == Counter(
        {"src/ops/rpu_eltwise_binary.cpp": 2}
    )
    assert _count_by_file(sources, r"\bCopyToDevice\b") == Counter(
        {"src/ops/rpu_eltwise_binary.cpp": 2}
    )
    for forbidden in (
        "CopyFromDevice",
        "CopyMemoryChannel",
        "CopyMemory",
        "CopyToDeviceBatch",
        "CopyFromDeviceBatch",
        "CopyMemoryBatch",
        "CopyMemoryBatchChannels",
        "RpuLaunchKernel",
    ):
        pattern = rf"\b{forbidden}\s*\("
        if forbidden == "CopyMemory":
            pattern = r"\bCopyMemory(?!Channel|Batch)\s*\("
        assert not _count_by_file(sources, pattern), forbidden
    for forbidden in (
        "CopyFromDevice",
        "CopyMemoryChannel",
        "CopyMemory",
        "CopyToDeviceBatch",
        "CopyFromDeviceBatch",
        "CopyMemoryBatch",
        "CopyMemoryBatchChannels",
        "RpuLaunchKernel",
    ):
        assert not _count_by_file(
            sources, rf"\b{forbidden}\b"
        ), forbidden


def test_direct_dma_guards_precede_every_side_effect() -> None:
    sources = _sources()
    memcpy = sources["src/ops/rpu_memcpy.cpp"]
    guard = "RpuKernelGraph::check_direct_immediate_dma_allowed"
    guarded_memcpy_paths = (
        "rpu_launch_ddr_broadcast_spm_dma_mutable",
        "rpu_launch_ddr_scatter_spm_dma_mutable",
        "rpu_launch_spm_scatter_ddr_dma_mutable",
        "rpu_launch_spm_copy_ddr_dma_mutable",
        "rpu_launch_spm_scatter_spm_dma",
        "rpu_launch_spm_local_shard_copy_dma",
        "rpu_launch_spm_gather_spm_dma",
        *IMMEDIATE_APIS,
        "rpu_launch_assembled_prefix_parts_dma_immediate",
    )
    sensitive = re.compile(
        r"\*\s*live_(?:src|dst)_base|rpu_ddr_flush\s*\(|RpuDdrFlush\s*\(|RpuGetDevAddr\s*\(|"
        r"QueueCache::instance\s*\(\s*\)\s*\.get\s*\(|"
        r"rpu_add_dma_checked\s*\(|(?:->|\.)\s*add_barrier\s*\(|"
        r"submit_immediate_batch\s*\("
        r"|\b(?:build_batch|enqueu_batch|discard_completed_batch)\s*\("
    )
    for name in guarded_memcpy_paths:
        _, begin, end = _function_span(memcpy, name)
        body = memcpy[begin:end]
        guard_pos = body.find(guard)
        side_effects = tuple(sensitive.finditer(body))
        assert guard_pos >= 0, name
        assert side_effects, name
        assert guard_pos < min(match.start() for match in side_effects), name

    kernel_cache = sources["src/core/rpu_kernel_cache.h"]
    _, begin, end = _function_span(kernel_cache, "rpu_add_dma_checked")
    add_body = kernel_cache[begin:end]
    assert add_body.index("add_dma_kernel") < add_body.index(
        "TORCH_CHECK(rc == 0"
    )

    _, begin, end = _function_span(memcpy, "submit_immediate_batch")
    submit_body = memcpy[begin:end]
    build_pos = submit_body.index("build_batch")
    build_check = re.search(
        r"TORCH_CHECK\s*\(\s*build_rc\s*==\s*0", submit_body
    )
    enqueue_pos = submit_body.index("enqueu_batch")
    enqueue_check = re.search(
        r"TORCH_CHECK\s*\(\s*enqueue_rc\s*==\s*0", submit_body
    )
    discard_pos = submit_body.index("discard_completed_batch")
    discard_check = re.search(
        r"TORCH_CHECK\s*\(\s*discard_rc\s*==\s*0", submit_body
    )
    assert build_check is not None
    assert enqueue_check is not None
    assert discard_check is not None
    assert (
        build_pos
        < build_check.start()
        < enqueue_pos
        < enqueue_check.start()
        < discard_pos
        < discard_check.start()
    )
    assert submit_body.count("build_batch") == 1
    assert submit_body.count("enqueu_batch") == 1
    assert submit_body.count("discard_completed_batch") == 1

    eltwise = sources["src/ops/rpu_eltwise_binary.cpp"]
    _, begin, end = _function_span(
        eltwise, "rpu_launch_eltwise_binary_sameshape_kernel_spm"
    )
    eltwise_body = eltwise[begin:end]
    guard_pos = eltwise_body.index(guard)
    assert not eltwise_body[:guard_pos].strip()
    eltwise_sensitive = re.compile(
        r"\bGET_KERNEL\s*\(|\bGET_QUEUE\s*\(|"
        r"\bLocalSPM_t\s+[A-Za-z_]\w*\s*[\(\{{]|"
        r"\brpu_ddr_flush\s*\(|\bCopyToDevice\s*\("
    )
    eltwise_side_effects = tuple(eltwise_sensitive.finditer(eltwise_body))
    assert eltwise_side_effects
    assert guard_pos < min(match.start() for match in eltwise_side_effects)
    assert eltwise_body.count("CopyToDevice") == 2
    assert eltwise_body.count("TORCH_CHECK(copy_") == 2

    execute = sources["src/graph/graph_runtime_execute.cpp"]
    _, begin, end = _function_span(
        execute, "RpuKernelGraph::prepare_segment_queue"
    )
    prepare_body = execute[begin:end]
    assert prepare_body.count("rpu_add_dma_mutable_checked") == 2
    assert prepare_body.count("rpu_add_dma_checked") == 1
    assert "add_dma_kernel(" not in prepare_body
    assert "add_dma_kernel_mutable(" not in prepare_body

    _, begin, end = _function_span(
        execute, "RpuKernelGraph::launch_segment_sync_only"
    )
    sync_body = execute[begin:end]
    update_calls = tuple(re.finditer(
        r"(?:->|\.)\s*update_dma_kernel\s*\(", sync_body
    ))
    update_check = re.search(
        r"if\s*\(\s*rc\s*!=\s*0\s*\)\s*\{\s*return\s+rc\s*;\s*\}",
        sync_body,
    )
    assert len(update_calls) == 1
    assert update_check is not None
    for admission in (
        "validate_semantic_dma_owner_for_execution(",
        "checked_dma_live_address(",
        "rpu_resolve_device_dma_endpoints(",
    ):
        assert sync_body.index(admission) < update_calls[0].start()
    assert update_calls[0].start() < update_check.start()
    assert update_check.end() < sync_body.index("enqueu_batch(")

    grouped = sources["src/ops/rpu_grouped_parallel_linear.cpp"]
    _, begin, end = _function_span(grouped, "rpu_grouped_parallel_linear_test")
    grouped_body = grouped[begin:end]
    passthrough = re.search(
        r"TORCH_CHECK\s*\(\s*RpuKernelGraph::active\(\)\.state\(\)\s*==\s*"
        r"RpuKernelGraph::State::PASSTHROUGH", grouped_body)
    assert passthrough is not None
    for effect in ("SPM_ALLOC.init()", "at::zeros(", "SPM_ALLOC.alloc_temporary_aliased("):
        assert passthrough.start() < grouped_body.index(effect)

    unary = sources["src/ops/rpu_eltwise_unary.cpp"]
    _, begin, end = _function_span(unary, "rpu_launch_eltwise_unary_kernel")
    unary_body = unary[begin:end]
    cold_guard = re.search(r"TORCH_CHECK\s*\(\s*!graph_dma::active\(\)", unary_body)
    assert cold_guard is not None
    for effect in ("GET_KERNEL(id)", "SPM_ALLOC.init()", "SPM_ALLOC.alloc_temporary(", BROADCAST):
        assert cold_guard.start() < unary_body.index(effect)

    _, begin, end = _function_span(unary, "rpu_gelu_erf")
    gelu_body = unary[begin:end]
    cold_guard = re.search(r"TORCH_CHECK\s*\(\s*!graph_dma::active\(\)", gelu_body)
    assert cold_guard is not None
    for effect in ("at::empty(", "SPM_ALLOC.init()", "SPM_ALLOC.alloc_temporary(", BROADCAST, SPM_COPY):
        assert cold_guard.start() < gelu_body.index(effect)

    _, begin, end = _function_span(execute, "RpuKernelGraph::execute_data_node")
    execute_body = execute[begin:end]
    assert execute_body.count("execute_ddr_copy(m)") == 1
    _, begin, end = _function_span(execute, "RpuKernelGraph::execute_ddr_copy")
    copy_body = execute[begin:end]
    assert copy_body.count("rpu_add_dma_checked(") == 1
    for operation, result in (
        ("build_batch()", "build_rc"),
        ("enqueu_batch(", "enqueue_rc"),
        ("discard_completed_batch()", "discard_rc"),
    ):
        assert copy_body.index(operation) < copy_body.index(f"TORCH_CHECK({result} == 0")
    assert "catch (...)" in copy_body
    assert "data_dma_queue_.reset()" in copy_body


def test_direct_and_caching_allocators_share_the_dma_endpoint_registry() -> None:
    sources = _sources()
    tensor_ops = sources["src/core/rpu_tensor_ops.inc"]
    direct_alloc = tensor_ops.index("rhino_lkn::RpuDdrAlloc(nbytes)")
    register = tensor_ops.index(
        "rpu_register_ddr_dma_allocation(ptr, nbytes)", direct_alloc
    )
    registered_deleter = tensor_ops.index(
        "&rpu_free_registered_ddr", register
    )
    assert direct_alloc < register < registered_deleter

    caching = sources["src/core/rpu_caching_allocator.cpp"]
    _, begin, end = _function_span(caching, "RPUCachingAllocator::alloc_block")
    allocation_body = caching[begin:end]
    assert (
        allocation_body.index("RpuDdrAlloc(alloc_size)")
        < allocation_body.index("rpu_register_ddr_dma_segment(ptr, alloc_size)")
        < allocation_body.index("return block;")
    )
    _, begin, end = _function_span(caching, "RPUCachingAllocator::malloc")
    malloc_body = caching[begin:end]
    logical_register = malloc_body.index(
        "rpu_register_ddr_dma_logical_allocation(block->ptr, orig_size)")
    assert logical_register < malloc_body.index("return block->ptr;")
    assert re.search(
        r"catch\s*\(\s*\.\.\.\s*\)\s*\{\s*free\(block->ptr\);\s*throw;\s*\}",
        malloc_body[logical_register:],
    )
    _, begin, end = _function_span(caching, "RPUCachingAllocator::free")
    free_body = caching[begin:end]
    retirement = re.search(
        r"if\s*\(\s*block->allocation_id\s*!=\s*0\s*&&\s*"
        r"!rpu_unregister_ddr_dma_logical_allocation\(\s*"
        r"ptr,\s*block->allocation_id\)\s*\)\s*\{[^{}]*return;\s*\}",
        free_body,
    )
    assert retirement is not None
    assert retirement.end() < free_body.index("block->allocated = false;")
    for reuse in ("free_block_fast(block, size_class)", "free_block(block)"):
        assert free_body.index("block->allocation_id = 0;") < free_body.index(reuse)
    for release in ("RPUCachingAllocator::release_block", "RPUCachingAllocator::emptyCache"):
        _, begin, end = _function_span(caching, release)
        release_body = caching[begin:end]
        assert release_body.index("rpu_unregister_ddr_dma_segment(block->ptr)") < (
            release_body.index("RpuDdrFree(block->ptr)"))
    _, begin, end = _function_span(
        caching, "rpu_unregister_ddr_dma_allocation"
    )
    unregister_body = caching[begin:end]
    assert unregister_body.index("rpu_unregister_ddr_dma_logical_allocation(") < (
        unregister_body.index("rpu_unregister_ddr_dma_segment(cpu_base)"))
    assert "RpuGetDevAddr" not in unregister_body
    assert "RpuGetHostddr" not in unregister_body




def test_immediate_dma_call_inventory_is_exact() -> None:
    sources = _sources()
    call_pattern = re.compile(
        rf"\b(?P<api>{'|'.join(map(re.escape, IMMEDIATE_APIS))})\s*\("
    )

    spans: dict[str, list[tuple[str, int, int]]] = {}
    actual: Counter[tuple[str, str, str]] = Counter()
    for (path, function), expected in IMMEDIATE_CALLERS.items():
        _, begin, end = _function_span(sources[path], function)
        spans.setdefault(path, []).append((function, begin, end))
        body = sources[path][begin:end]
        observed = Counter(match.group("api") for match in call_pattern.finditer(body))
        assert observed == expected, (path, function, observed, expected)
        actual.update(
            {(path, function, api): count for api, count in observed.items()}
        )

    definition_positions = {
        ("src/ops/rpu_memcpy.cpp", _function_span(
            sources["src/ops/rpu_memcpy.cpp"], api
        )[0])
        for api in IMMEDIATE_APIS
    }
    declarations: Counter[tuple[str, str]] = Counter()
    for path, source in sources.items():
        for match in call_pattern.finditer(source):
            if (path, match.start()) in definition_positions:
                continue
            owners = [
                function
                for function, begin, end in spans.get(path, ())
                if begin <= match.start() < end
            ]
            if len(owners) == 1:
                continue
            if (
                path == "src/core/rpu_kernel_decls.h"
                and _is_function_declaration(source, match.start())
            ):
                declarations[(path, match.group("api"))] += 1
                continue
            assert False, (path, match.start(), match.group("api"), owners)

    assert declarations == Counter(
        {
            ("src/core/rpu_kernel_decls.h", api): 1
            for api in IMMEDIATE_APIS
        }
    )

    for api in IMMEDIATE_APIS:
        expected_bare = Counter(
            {
                "src/core/rpu_kernel_decls.h": 1,
                "src/ops/rpu_memcpy.cpp": 1,
            }
        )
        for (path, _function), expected in IMMEDIATE_CALLERS.items():
            expected_bare[path] += expected[api]
        assert _count_by_file(sources, rf"\b{re.escape(api)}\b") == (
            +expected_bare
        ), api

    assert sum(actual.values()) == 84
    assert len(IMMEDIATE_CALLERS) == 39
    assert Counter(
        {
            api: sum(
                count
                for (_path, _function, observed_api), count in actual.items()
                if observed_api == api
            )
            for api in IMMEDIATE_APIS
        }
        ) == Counter(
        {
            BROADCAST: 35,
            SPM_COPY: 32,
            BYTE_COPY: 2,
            DDR_SCATTER: 9,
            SPM_SCATTER: 6,
        }
    )


def test_assembled_prefix_batch_is_internal_and_failure_cleanup_is_bounded():
    sources = _sources()
    helper = "rpu_launch_assembled_prefix_parts_dma_immediate"
    assert _count_by_file(sources, rf"\b{helper}\b") == Counter(
        {"src/ops/rpu_embedding.cpp": 2, "src/ops/rpu_memcpy.cpp": 1})
    _, begin, end = _function_span(sources["src/ops/rpu_embedding.cpp"], "rpu_assemble_inputs_embeds_parts")
    operator = sources["src/ops/rpu_embedding.cpp"][begin:end]
    prefix_begin = operator.rindex("if (prefix_is_embeddings)")
    prefix_branch = operator[prefix_begin:operator.index("} else {", prefix_begin)]
    assert prefix_branch.count(helper) == 1 and prefix_branch.index(helper) < prefix_branch.index("return out;")
    _, begin, end = _function_span(sources["src/ops/rpu_memcpy.cpp"], helper)
    body = sources["src/ops/rpu_memcpy.cpp"][begin:end]
    assert body.count("submit_immediate_batch(") == 1
    assert body.index("catch (...)") < body.index("QueueCache::instance().clear()") < body.index("throw;")
    assert not re.search(r"\b(?:rpu_ddr_flush_force|set_ddr_flush|set_ddr_flush_force)\b", body)
