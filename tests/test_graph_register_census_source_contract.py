"""Host-only source contract for the GR00T Z2 DDR-register census gate."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
GRAPH_RUNTIME_SOURCES = (
    "src/graph/graph_runtime.h",
    "src/graph/graph_runtime.cpp",
    "src/graph/graph_register_census.cpp",
    "src/graph/graph_runtime_execute.cpp",
)
DIRECT_DMA_SOURCE_CONTRACT = "tests/test_graph_direct_dma_source_contract.py"
REGISTER_CENSUS_SOURCE_CONTRACT = (
    "tests/test_graph_register_census_source_contract.py"
)
FNV64_OFFSET = 14695981039346656037
FNV64_PRIME = 1099511628211
FNV64_MASK = (1 << 64) - 1


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _blank_preserving_newlines(value: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in value)


_RAW_STRING = re.compile(
    r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"',
    re.DOTALL,
)


def _strip_cpp_comments_and_literals(source: str) -> str:
    """Blank comments and literals without changing offsets or line numbers."""

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
    return "".join(out)


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


def _function_body(relative: str, name: str) -> str:
    source = _strip_cpp_comments_and_literals(_read(relative))
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
        return source[brace + 1 : _match_close(source, brace, "{", "}")]
    raise AssertionError(f"definition not found: {relative}:{name}")


def _function_body_raw(relative: str, name: str) -> str:
    raw = _read(relative)
    source = _strip_cpp_comments_and_literals(raw)
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
        close = _match_close(source, brace, "{", "}")
        return raw[brace + 1 : close]
    raise AssertionError(f"definition not found: {relative}:{name}")


def _overload_body(relative: str, name: str, signature_token: str) -> str:
    """Return the unique overload whose declaration contains ``signature_token``."""

    source = _strip_cpp_comments_and_literals(_read(relative))
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(")
    matches: list[str] = []
    for match in pattern.finditer(source):
        open_paren = source.find("(", match.start())
        close_paren = _match_close(source, open_paren, "(", ")")
        if signature_token not in source[match.start() : close_paren + 1]:
            continue
        cursor = close_paren + 1
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        brace = source.find("{", cursor)
        semicolon = source.find(";", cursor)
        if brace < 0 or (semicolon >= 0 and semicolon < brace):
            continue
        matches.append(source[brace + 1 : _match_close(source, brace, "{", "}")])
    assert len(matches) == 1, (
        f"expected one overload: {relative}:{name}:{signature_token}; "
        f"found {len(matches)}"
    )
    return matches[0]


def _overload_body_raw(relative: str, name: str, signature_token: str) -> str:
    """Raw-text counterpart of ``_overload_body`` for literal assertions."""

    raw = _read(relative)
    source = _strip_cpp_comments_and_literals(raw)
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(")
    matches: list[str] = []
    for match in pattern.finditer(source):
        open_paren = source.find("(", match.start())
        close_paren = _match_close(source, open_paren, "(", ")")
        if signature_token not in source[match.start() : close_paren + 1]:
            continue
        cursor = close_paren + 1
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        brace = source.find("{", cursor)
        semicolon = source.find(";", cursor)
        if brace < 0 or (semicolon >= 0 and semicolon < brace):
            continue
        close = _match_close(source, brace, "{", "}")
        matches.append(raw[brace + 1 : close])
    assert len(matches) == 1, (
        f"expected one raw overload: {relative}:{name}:{signature_token}; "
        f"found {len(matches)}"
    )
    return matches[0]


def _cpp_block(relative: str, anchor: str) -> str:
    source = _strip_cpp_comments_and_literals(_read(relative))
    cursor = 0
    while True:
        start = source.find(anchor, cursor)
        if start < 0:
            raise AssertionError(f"definition not found: {relative}:{anchor}")
        brace = source.find("{", start + len(anchor))
        semicolon = source.find(";", start + len(anchor))
        if brace >= 0 and (semicolon < 0 or brace < semicolon):
            close = _match_close(source, brace, "{", "}")
            return source[brace + 1 : close]
        cursor = start + len(anchor)


def _squash(source: str) -> str:
    return " ".join(source.split())


def _fnv_graph_node_kinds(kinds: list[int]) -> int:
    value = FNV64_OFFSET
    for kind in kinds:
        value = ((value ^ kind) * FNV64_PRIME) & FNV64_MASK
    return value


def _graph_dma_burst(channel_count: int) -> list[int]:
    assert 1 <= channel_count <= 8
    dma = 8
    barrier = 9
    return (
        [barrier] * (channel_count - 1)
        + [dma] * channel_count
        + [barrier] * (channel_count - 1)
    )


def _census_escape_gate(source: str) -> tuple[int, int]:
    active = source.index("!kernel_register_census_active_")
    start = source.rfind("TORCH_CHECK", 0, active)
    assert start >= 0
    open_paren = source.index("(", start)
    close_paren = _match_close(source, open_paren, "(", ")")
    gate = source[start : close_paren + 1]
    for state in (
        "!kernel_register_census_active_",
        "!has_kernel_register_census_nodes_",
        "!pending_kernel_register_census_.has_value()",
    ):
        assert state in gate
    return start, close_paren + 1


def _block_after(source: str, anchor: str) -> tuple[str, int]:
    start = source.index(anchor)
    brace = source.index("{", start + len(anchor))
    close = _match_close(source, brace, "{", "}")
    return source[brace + 1 : close], close + 1


def _fp16_linear_body() -> str:
    body = _function_body(
        "src/ops/rpu_linear.cpp",
        "rpu_launch_linear_spm_to_spm_acc16_kernel",
    )
    _, fp16_start = _block_after(
        body, "if (weight.scalar_type() == at::kByte)"
    )
    return body[fp16_start:]


def _w8_linear_body() -> str:
    body = _function_body(
        "src/ops/rpu_linear.cpp",
        "rpu_launch_linear_spm_to_spm_acc16_kernel",
    )
    branch, _ = _block_after(
        body, "if (weight.scalar_type() == at::kChar || fp8_e4m3_weight)"
    )
    return branch


def _kv_function_body(kind: str) -> str:
    return _function_body(
        "src/ops/rpu_llama_kvcache.cpp",
        f"rpu_launch_insert_{kind}cache_spm_unified_with_plan",
    )


def _kv_launch_once_body(kind: str) -> str:
    body = _kv_function_body(kind)
    launch, _ = _block_after(body, "auto launch_once =")
    return launch


def _kv_route_body(kind: str) -> str:
    body = _kv_function_body(kind)
    _, route_start = _block_after(body, "auto launch_once =")
    return body[route_start:]


def _sdpa_body() -> str:
    return _function_body(
        "src/ops/rpu_sdpa.cpp",
        "launch_sdpa_spm_unified_impl",
    )


def _allreduce_body() -> str:
    return _function_body(
        "src/core/rpu_runtime_extras.cpp",
        "rpu_launch_all_reduce_sum_residual_impl",
    )


def _allreduce_ring_body() -> str:
    return _function_body(
        "src/core/rpu_runtime_extras.cpp",
        "launch_ring_all_reduce_sum_residual",
    )


def _positions(source: str, pattern: str) -> list[int]:
    return [match.start() for match in re.finditer(pattern, source)]


def _assert_ddr_launch_order(label: str, source: str) -> None:
    stages = _positions(source, r"\bstage_kernel_ddr_registers\s*\(")
    gets = _positions(source, r"\bGET_KERNEL\s*\(")
    flushes = _positions(source, r"\brpu_ddr_flush\s*\(")
    resets = _positions(source, r"(?:->|\.)\s*reset_regs\s*\(")
    writes = _positions(
        source,
        r"\b[A-Za-z_]\w*\s*\.\s*write\s*\(\s*\*\s*[A-Za-z_]\w*\s*\)",
    )
    enqueues = _positions(source, r"->\s*enqueu_kernel\s*\(")

    assert len(stages) == 1, f"{label}: expected one typed DDR stage"
    assert len(gets) == 1, f"{label}: expected one final Graph-aware GET"
    assert flushes, f"{label}: DDR flush must remain explicit"
    assert len(writes) == 1, f"{label}: staged address regs need one writer"
    assert len(enqueues) == 1, f"{label}: expected one kernel enqueue"

    stage = stages[0]
    get = gets[0]
    write = writes[0]
    enqueue = enqueues[0]
    assert stage < min(flushes) <= max(flushes) < get, label
    assert get < write < enqueue, label
    assert all(stage < reset < write for reset in resets), label
    after_writer = source[write:enqueue]
    forbidden_after_writer = re.compile(
        r"\b(?:set_regs|reset_regs|setup\w*regs|patch\w*|"
        r"update_\w*register\w*)\s*\("
        r"|\bRegisterPatch\b"
    )
    assert not forbidden_after_writer.search(after_writer), label


def _assert_writer_owns_registers(
    label: str, source: str, registers: tuple[int, ...]
) -> None:
    for register in registers:
        direct_write = re.compile(
            rf"\bset_regs\s*\(\s*{register}\s*,"
        )
        assert not direct_write.search(source), (
            f"{label}: DDR address register {register} bypasses "
            "GraphKernelRegisterWriter::write"
        )


def test_typed_ddr_hooks_precede_flush_get_reset_write_and_enqueue():
    admitted = {
        "fp16 linear": _fp16_linear_body(),
        "W8A16 linear": _w8_linear_body(),
        "K-cache insert": _kv_launch_once_body("k"),
        "V-cache insert": _kv_launch_once_body("v"),
        "SPM SDPA": _sdpa_body(),
    }
    for label, source in admitted.items():
        _assert_ddr_launch_order(label, source)


def test_denied_linear_routes_stop_before_side_effects():
    linear = _function_body(
        "src/ops/rpu_linear.cpp",
        "rpu_launch_linear_spm_to_spm_acc16_kernel",
    )
    linear_guard, linear_guard_end = _block_after(
        linear, "if (register_census_active)"
    )
    assert "weight.scalar_type() == at::kHalf" in linear_guard
    assert "weight.scalar_type() == at::kChar" in linear_guard
    assert "fp8_e4m3_weight" in linear_guard
    assert linear_guard_end < linear.index("if (use_acc32 &&")
    assert linear_guard_end < linear.index(
        "if (weight.scalar_type() == at::kChar || fp8_e4m3_weight)"
    )
    assert linear_guard_end < linear.index(
        "if (weight.scalar_type() == at::kByte)"
    )

def test_w8_linear_stages_exact_weight_and_scale_abi():
    w8 = _w8_linear_body()
    stage = w8.index("stage_kernel_ddr_registers(")
    assert w8.count("GraphDdrRegisterOperandSpec{") == 2
    assert re.search(
        r"GraphDdrRegisterRole::Int8ModelWeight,[\s\S]*?"
        r"GraphDdrRegisterAccess::Read,[\s\S]*?"
        r"GraphDdrRegisterEncoding::DevAddrShift8LoHi,[\s\S]*?2, 3",
        w8,
    )
    assert "GraphDdrRegisterRole::Fp8ModelWeight" in w8
    assert "static_cast<uint16_t>(is_fp8_e4m3 ? 5 : 1)" in w8
    assert re.search(
        r"GraphDdrRegisterRole::PerChannelScale,[\s\S]*?"
        r"GraphDdrRegisterAccess::Read,[\s\S]*?"
        r"GraphDdrRegisterEncoding::DevAddrShift8LoHi,[\s\S]*?"
        r"20, 21",
        w8,
    )
    for validation in (
        "weight.size(0) == N",
        "weight.size(1) == K",
        "weight.is_contiguous()",
        "scale.dim() == 1",
        "scale.size(0) == N",
        "scale.is_contiguous()",
        "weight.nbytes()",
        "scale.nbytes()",
    ):
        assert w8.index(validation) < stage
    _assert_writer_owns_registers(
        "W8A16 linear", w8, (2, 3, 20, 21)
    )


def test_kv_multilaunch_plan_is_validated_and_census_allows_each_node():
    for kind in ("k", "v"):
        body = _kv_function_body(kind)
        validation = body.index("rpu_validate_kvinsert_segment_plan(")
        launch_definition = body.index("auto launch_once =")
        assert validation < launch_definition
        assert "stage_kernel_ddr_registers(" not in body[:launch_definition]
        assert "rpu_ddr_flush(" not in body[:launch_definition]
        assert "RpuGetDevAddr(" not in body[:launch_definition]
        assert "GET_KERNEL(" not in body[:launch_definition]
        assert "std::getenv(" not in body
        assert "rpu_resolve_kvinsert_segment_plan(" not in body
        assert "rpu_resolve_legacy_kvinsert_segment_plan(" not in body

        route = _kv_route_body(kind)
        assert "single-launch" not in route
        first_dispatch = route.index("launch_once(part,")
        # The first pass checks every segment address.  Only the second pass can
        # stage a node, so an invalid later segment cannot leave a partial graph.
        assert route[:first_dispatch].count("plan.segment_count()") >= 2
        for side_effect in (
            "stage_kernel_ddr_registers(",
            "rpu_ddr_flush(",
            "RpuGetDevAddr(",
            "GET_KERNEL(",
            "launch_once(",
        ):
            assert side_effect not in route[:first_dispatch]

    # The graph runtime counts rule occurrences and compares the next cursor
    # node; it does not require a unique KernelId or cache operand per rule.
    stage = _function_body(
        "src/graph/graph_register_census.cpp",
        "RpuKernelGraph::stage_kernel_ddr_registers",
    )
    consume = _function_body(
        "src/graph/graph_register_census.cpp",
        "RpuKernelGraph::consume_pending_kernel_register_census",
    )
    assert "nodes_[cursor_]" in stage
    assert "check_same_occurrence(" in stage
    assert "++kernel_register_census_rule_counts_[rule_index]" in consume


def test_ddr_address_register_pairs_are_owned_only_by_the_typed_writer():
    admitted = {
        "fp16 linear": (_fp16_linear_body(), (2, 3)),
        "K-cache insert": (_kv_function_body("k"), (8, 9)),
        "V-cache insert": (_kv_function_body("v"), (8, 9)),
        "SPM SDPA": (_sdpa_body(), (50, 51, 52, 53)),
    }
    for label, (source, registers) in admitted.items():
        _assert_writer_owns_registers(label, source, registers)

    graph_sources = "\n".join(
        _read(relative) for relative in GRAPH_RUNTIME_SOURCES
    )
    assert "GraphKernelRegisterWriter::write" in graph_sources
    definition_file = next(
        relative
        for relative in GRAPH_RUNTIME_SOURCES
        if "GraphKernelRegisterWriter::write" in _read(relative)
    )
    writer = _function_body(
        definition_file, "GraphKernelRegisterWriter::write"
    )
    assert "write_register_pairs(kernel, pairs_" in writer
    pair_writer = _function_body(
        definition_file, "write_register_pairs"
    )
    assert pair_writer.count("kernel.set_regs(") == 2
    assert pair_writer.count("::rhino_lkn::kKernelRegOk") == 2


def test_ring_allreduce_is_staged_before_lookup_and_enqueue():
    source = _allreduce_ring_body()
    stages = _positions(source, r"\bstage_kernel_no_ddr\s*\(")
    gets = _positions(source, r"\bGET_KERNEL\s*\(")
    resets = _positions(source, r"(?:->|\.)\s*reset_regs\s*\(")
    enqueues = _positions(source, r"->\s*enqueu_kernel\s*\(")
    resolves = _positions(source, r"\bRpuGetDevAddr\s*\(")

    assert len(stages) == len(gets) == len(enqueues) == 1
    assert resolves and stages[0] < gets[0] < resolves[0] < enqueues[0]
    assert any(gets[0] < reset < enqueues[0] for reset in resets)
    assert "GraphKernelNoDdrProof::RingAllReduceSpmResidual" in source


def test_allreduce_validates_before_ring_launch():
    source = _allreduce_body()
    core_gate = source.index("input_num_cores >= 1")
    range_gate = source.index("ring_all_reduce_sum_residual_admitted(")
    launch = source.index("launch_ring_all_reduce_sum_residual(")

    assert core_gate < range_gate < launch
    assert "rpu_launch_fill_spm_kernel(" not in source
    assert "GET_KERNEL(" not in source
    assert "RpuGetDevAddr(" not in source
    assert source.count("rpu_launch_all_reduce_sum_residual_impl(") == 0


def test_partial_allreduce_preparation_zeroes_all_cores_before_the_producer():
    source = _function_body(
        "src/core/rpu_runtime_extras.cpp",
        "rpu_prepare_ring_all_reduce_input",
    )
    for token in (
        "input_num_cores <= output_num_cores",
        "output_num_cores <= kRingCoreNum",
        "input_num_cores == output_num_cores",
        "kFillAlignmentElements == 128",
        "rpu_launch_fill_spm_kernel(",
        "input_spm",
        "c10::Half(0.0f)",
        "kRingCoreNum",
        "input_num_cores",
    ):
        assert token in source
    # Every active consumer lane must be zero before the partial producer runs;
    # the explicit 4 -> 6 path must neither leave lanes 4/5 dirty nor touch 6/7.
    # The production wrapper host recorder observes the actual launch counts
    # in tests/test_core_count_wrappers.py.
    fill = source[source.index("rpu_launch_fill_spm_kernel("):]
    assert re.search(
        r"c10::Half\(0\.0f\)\s*,\s*output_num_cores\s*\);", fill
    )
    assert source.index("input_num_cores == output_num_cores") < source.index(
        "rpu_launch_fill_spm_kernel("
    )


def test_host_fixture_freezes_fnv_node_kinds_and_producer_dma_bursts():
    runtime = _read("src/graph/graph_register_census.cpp")
    assert (
        "constexpr uint64_t kFnvOffset = "
        "UINT64_C(14695981039346656037);"
    ) in runtime
    assert (
        "constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);"
        in runtime
    )

    enum_body = _cpp_block("src/graph/graph_runtime.h", "enum class GraphNodeKind")
    enum_members = tuple(
        member.split("=", 1)[0].strip()
        for member in enum_body.split(",") if member.strip()
    )
    assert enum_members == (
        "Kernel",
        "Memcpy",
        "Memset",
        "ChildGraph",
        "Branch",
        "HostCallback",
        "Tier3Oneshot",
        "Dma",
        "Barrier",
    )
    kernel, dma, barrier = 0, 8, 9
    assert "Kernel = 0" in enum_body
    assert "Dma = 8" in enum_body
    assert "Barrier = 9" in enum_body
    assert _fnv_graph_node_kinds([kernel, dma, barrier]) == int(
        "d931e9186bf82254", 16
    )

    assert _graph_dma_burst(8) == [barrier] * 7 + [dma] * 8 + [barrier] * 7
    assert _graph_dma_burst(1) == [dma]
    producer_dma_nodes = _graph_dma_burst(8) * 36 + _graph_dma_burst(1) * 8
    assert producer_dma_nodes.count(dma) == 296
    assert producer_dma_nodes.count(barrier) == 504
    # The original 74 Kernel positions come from the real trace.  Replacing
    # eight scatter+gather pairs with eight ring kernels removes eight kernel
    # nodes; burst reconstruction remains unchanged.
    assert len(producer_dma_nodes) == 800
    assert len(producer_dma_nodes) + 66 == 866

    policy = _squash(
        _function_body_raw(
            "src/fused/rpu_gr00t_spm_z2.cpp",
            "gr00t_z2_register_policy",
        )
    )
    assert "already-passing captured BUILD traces" in policy
    assert ("/" + "-".join(("nfs3", "sh")) + "/") not in policy
    assert "Barrier×7,Dma×8,Barrier×7" in policy
    assert "Kernel=0,Dma=8,Barrier=9" in policy
    assert "866, 296, 504, 66" in policy
    assert "UINT64_C(0xb07b80c16deb5555)" in policy


def test_gr00t_z2_policy_freezes_every_rule_count_hash_and_phase():
    policy = _squash(
        _function_body_raw(
            "src/fused/rpu_gr00t_spm_z2.cpp",
            "gr00t_z2_register_policy",
        )
    )

    assert 'value.fingerprint = UINT64_C(0x475230305a325634);' in policy
    assert 'value.name = "gr00t_z2_register_census_v4";' in policy

    abi_specs = (
        "const auto weight = ddr_abi( "
        "GraphDdrRegisterRole::ModelWeight, "
        "GraphDdrRegisterAccess::Read, 2, 3);",
        "const auto key_write = ddr_abi( "
        "GraphDdrRegisterRole::KeyCache, "
        "GraphDdrRegisterAccess::Write, 8, 9);",
        "const auto value_write = ddr_abi( "
        "GraphDdrRegisterRole::ValueCache, "
        "GraphDdrRegisterAccess::Write, 8, 9);",
        "const auto key_read = ddr_abi( "
        "GraphDdrRegisterRole::KeyCache, "
        "GraphDdrRegisterAccess::Read, 50, 51);",
        "const auto value_read = ddr_abi( "
        "GraphDdrRegisterRole::ValueCache, "
        "GraphDdrRegisterAccess::Read, 52, 53);",
    )
    for abi in abi_specs:
        assert abi in policy

    exact_rules = (
        "typed_rule(KernelId::PL_AT_FP16_M384N96, 942, {weight})",
        "typed_rule(KernelId::PL_AT_FP16_M320N128, 12, {weight})",
        "typed_rule(KernelId::LLAMA_INSERT_KCACHE_V16, 128, {key_write})",
        "typed_rule(KernelId::LLAMA_INSERT_VCACHE_V16, 128, {value_write})",
        "typed_rule(KernelId::LLAMA_INSERT_KCACHE, 4, {key_write})",
        "typed_rule(KernelId::LLAMA_INSERT_VCACHE, 4, {value_write})",
        "typed_rule(KernelId::SDPA_FLASH_ATTN_SPM, 132, "
        "{key_read, value_read})",
        "explicit_no_ddr_rule( "
        "KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED, 264, "
        "GraphKernelNoDdrProof::RingAllReduceSpmResidual)",
        "explicit_no_ddr_rule( "
        "KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE, 128, "
        "GraphKernelNoDdrProof::RingAllReduceSpmResidual)",
        "intrinsic_no_ddr_rule(KernelId::MEMSET_SPM_MULTI_CORE_V2, 328)",
        "intrinsic_no_ddr_rule(KernelId::LAYER_NORM, 269)",
        "intrinsic_no_ddr_rule(KernelId::BINARY_SCALAR, 137)",
        "intrinsic_no_ddr_rule(KernelId::UNARY_GELU_TANH_SPM, 132)",
        "intrinsic_no_ddr_rule(KernelId::BINARY_SAMESHAPE, 12)",
        "intrinsic_no_ddr_rule(KernelId::UNARY_SPM, 4)",
    )
    rule_positions = [policy.index(rule) for rule in exact_rules]
    assert rule_positions == sorted(rule_positions)
    assert policy.count("typed_rule(") == 7
    assert policy.count("explicit_no_ddr_rule(") == 2
    assert policy.count("intrinsic_no_ddr_rule(") == 6

    dynamic_rule = (
        'GraphKernelRegisterRule dynamic_binary; '
        'dynamic_binary.kernel_name = "binary_1xC_NxC_tileC"; '
        "dynamic_binary.kind = "
        "GraphKernelRegisterRuleKind::IntrinsicNoDdr; "
        "dynamic_binary.expected_count = 1; "
        "value.rules.push_back(std::move(dynamic_binary));"
    )
    assert dynamic_rule in policy
    assert rule_positions[-1] < policy.index(dynamic_rule)
    assert policy.count("value.rules.push_back(") == 1
    assert "LLAMA_GEMV" not in policy
    assert "gemv_weight" not in policy

    exact_totals = {
        "expected_node_count": "10782",
        "expected_dma_count": "3033",
        "expected_barrier_count": "5124",
        "expected_kernel_count": "2625",
        "expected_typed_kernel_count": "1350",
        "expected_no_ddr_kernel_count": "1275",
        "expected_operand_count": "1482",
        "expected_weight_operand_count": "954",
        "expected_key_cache_operand_count": "264",
        "expected_value_cache_operand_count": "264",
        "expected_semantic_input_operand_count": "0",
        "expected_ordered_node_kind_hash": (
            "UINT64_C(0x1cfe991b4c0a51a5)"
        ),
        "expected_ordered_kernel_hash": (
            "UINT64_C(0x02a89226f052e980)"
        ),
    }
    for field, value in exact_totals.items():
        assert policy.count(f"value.{field} = {value};") == 1

    exact_phases = (
        "GraphKernelRegisterPhaseExpectation{ \"vl_encoder\", "
        "phase_stats( 866, 296, 504, 66, 36, 30, 40, 24, 8, 8, "
        "UINT64_C(0xb07b80c16deb5555), "
        "UINT64_C(0x728c2112d7fcf9b9))}",
        "GraphKernelRegisterPhaseExpectation{ \"denoise\", "
        "phase_stats( 9916, 2737, 4620, 2559, 1314, 1245, "
        "1442, 930, 256, 256, UINT64_C(0x72c4af3f08d45a75), "
        "UINT64_C(0xee5248a5e490c4e4))}",
    )
    phase_positions = [policy.index(phase) for phase in exact_phases]
    assert phase_positions == sorted(phase_positions)
    assert len(re.findall(r"GraphKernelRegisterPhaseExpectation\s*\{", policy)) == 2


def test_runtime_rejects_forced_oneshot_before_mutation_and_matches_policy_deeply():
    begin = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::begin_kernel_register_census",
        )
    )
    force_oneshot = begin.index("force_oneshot_on_replay_enabled()")
    digest_value = begin.index("const uint64_t policy_digest")
    stale_invocation_states = (
        "!kernel_register_census_poisoned_",
        "!kernel_register_census_complete_",
        "!kernel_register_census_invocation_armed_",
    )
    for stale_invocation_state in stale_invocation_states:
        assert begin.index(stale_invocation_state) < digest_value
    assert "kernel_register_census_poisoned_ = false" not in begin
    first_policy_store = begin.index("kernel_register_census_policy_ = policy")
    census_mutations = _positions(
        begin,
        r"\b(?:kernel_register_census_[A-Za-z0-9_]*|"
        r"has_kernel_register_census_nodes_|"
        r"next_kernel_register_writer_ticket_)\s*=(?!=)",
    )
    assert census_mutations
    candidate_authority = begin.index(
        "has_kernel_register_census_nodes_ = true"
    )
    assert force_oneshot < digest_value < candidate_authority
    assert min(census_mutations) == candidate_authority
    for stale_invocation_state in stale_invocation_states:
        assert begin.index(stale_invocation_state) < min(census_mutations)
    assert candidate_authority < first_policy_store
    for replay_identity in (
        "has_kernel_register_census_nodes_",
        "kernel_register_census_build_committed_",
        "kernel_register_census_policy_digest_ ==",
        "policy_digest",
        "same_policy(kernel_register_census_policy_, policy)",
        "kernel_register_census_plan_hash_ == physical_plan_hash",
    ):
        assert replay_identity in begin[digest_value:]

    digest = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp", "stable_policy_digest"
        )
    )
    equality = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp", "same_policy"
        )
    )
    policy_fields = (
        "fingerprint",
        "name",
        "expected_kernel_count",
        "expected_typed_kernel_count",
        "expected_no_ddr_kernel_count",
        "expected_operand_count",
        "expected_weight_operand_count",
        "expected_key_cache_operand_count",
        "expected_value_cache_operand_count",
        "expected_semantic_input_operand_count",
        "expected_node_count",
        "expected_dma_count",
        "expected_barrier_count",
        "expected_ordered_node_kind_hash",
        "expected_ordered_kernel_hash",
        "rules.size()",
        "phases.size()",
    )
    for field in policy_fields:
        assert f"policy.{field}" in digest
        assert f"lhs.{field}" in equality
        assert f"rhs.{field}" in equality

    for field in (
        "kernel_id",
        "kernel_name",
        "kind",
        "no_ddr_proof",
        "expected_count",
        "operands.size()",
    ):
        assert f"rule.{field}" in digest
    for field in ("role", "access", "encoding", "reg_lo", "reg_hi"):
        assert f"abi.{field}" in digest
    assert "phase.name" in digest
    assert "hash_stats(hash, phase.stats)" in digest

    for exact_rule_field in (
        "a.kernel_id != b.kernel_id",
        "a.kernel_name != b.kernel_name",
        "a.kind != b.kind",
        "a.no_ddr_proof != b.no_ddr_proof",
        "a.operands != b.operands",
        "a.expected_count != b.expected_count",
    ):
        assert exact_rule_field in equality
    assert "lhs.phases[i].name != rhs.phases[i].name" in equality
    assert "!same_stats(lhs.phases[i].stats, rhs.phases[i].stats)" in equality

    stats_fields = (
        "node_count",
        "dma_count",
        "barrier_count",
        "kernel_count",
        "typed_kernel_count",
        "no_ddr_kernel_count",
        "operand_count",
        "weight_operand_count",
        "key_cache_operand_count",
        "value_cache_operand_count",
        "semantic_input_operand_count",
        "ordered_node_kind_hash",
        "ordered_kernel_hash",
    )
    hash_stats = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp", "hash_stats"
        )
    )
    same_stats = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp", "same_stats"
        )
    )
    for field in stats_fields:
        assert f"stats.{field}" in hash_stats
        assert f"lhs.{field}" in same_stats
        assert f"rhs.{field}" in same_stats


def test_schema5_topology_binds_policy_and_plan_but_not_live_epoch():
    topology = _squash(
        _function_body(
            "src/graph/graph_runtime.cpp",
            "RpuKernelGraph::topology_hash_for_window",
        )
    )
    schema5, _ = _block_after(topology, "if (dma_topology_schema >= 5)")
    exact_prefix = (
        "hash.u64(kernel_register_census_policy_.fingerprint); "
        "hash.u64(kernel_register_census_policy_digest_); "
        "hash.u64(kernel_register_census_plan_hash_);"
    )
    assert exact_prefix in schema5
    assert "kernel_register_census_live_epoch_" not in topology


def test_typed_stage_rejects_policy_abi_and_replay_drift_before_snapshot():
    stage = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::stage_kernel_ddr_registers",
        )
    )
    active_gate = stage.index("!kernel_register_census_poisoned_")
    rule_lookup = stage.index("kernel_register_rule_for(id)", active_gate)
    kind_arity = stage.index(
        "rule.kind == GraphKernelRegisterRuleKind::TypedDdr", rule_lookup
    )
    abi_order = stage.index(
        "rule.operands[i] == operands[i].abi", kind_arity
    )
    replay_state = stage.index("if (state_ == State::REPLAYING)", abi_order)
    replay_cursor = stage.index("cursor_ < nodes_.size()", replay_state)
    replay_identity = stage.index(
        "node.kernel_id.has_value()", replay_cursor
    )
    replay_census = stage.index(
        "built_census->kind == GraphKernelRegisterCensus::Kind::TypedDdr",
        replay_identity,
    )
    replay_abi = stage.index(
        "built_census->operands[i].abi == operands[i].abi", replay_census
    )
    active_snapshot = stage.index("snapshot_operand(", active_gate)
    pending_publish = stage.index(
        "pending_kernel_register_census_.emplace(", active_snapshot
    )
    assert (
        active_gate
        < rule_lookup
        < kind_arity
        < abi_order
        < replay_state
        < replay_cursor
        < replay_identity
        < replay_census
        < replay_abi
        < active_snapshot
        < pending_publish
    )
    assert "RpuGetDevAddr(" not in stage


def test_register_occurrences_retain_strong_tensor_and_storage_ownership():
    declaration = _cpp_block(
        "src/graph/graph_runtime.h", "struct GraphDdrRegisterOccurrence"
    )
    for field in (
        "at::Tensor tensor_owner;",
        "c10::Storage storage_owner;",
        "const c10::TensorImpl* tensor_impl",
        "const c10::StorageImpl* storage_impl",
        "const void* data_ptr",
        "const void* storage_data_ptr",
        "at::ScalarType dtype",
        "c10::Device device",
        "c10::Layout layout",
        "std::vector<int64_t> sizes",
        "std::vector<int64_t> strides",
        "int64_t storage_offset",
        "uint64_t tensor_topology_id",
        "size_t tensor_nbytes",
        "size_t storage_nbytes",
        "uint64_t dev_addr",
    ):
        assert field in declaration

    snapshot = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp", "snapshot_operand"
        )
    )
    assignments = (
        "occurrence.tensor_owner = spec.tensor",
        "occurrence.storage_owner = spec.tensor.storage()",
        "occurrence.tensor_impl = spec.tensor.unsafeGetTensorImpl()",
        "occurrence.storage_impl = occurrence.storage_owner.unsafeGetStorageImpl()",
        "occurrence.data_ptr = spec.tensor.const_data_ptr()",
        "occurrence.storage_data_ptr = occurrence.storage_owner.data()",
        "occurrence.tensor_topology_id = tensor_topology_id(",
        "occurrence.dev_addr = rhino_lkn::RpuGetDevAddr(",
    )
    positions = [snapshot.index(assignment) for assignment in assignments]
    assert positions == sorted(positions)

    exact = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp", "check_same_occurrence"
        )
    )
    exact_fields = (
        "abi",
        "tensor_impl",
        "storage_impl",
        "data_ptr",
        "storage_data_ptr",
        "dtype",
        "device",
        "layout",
        "sizes",
        "strides",
        "storage_offset",
        "tensor_nbytes",
        "storage_nbytes",
        "dev_addr",
        "tensor_topology_id",
    )
    for field in exact_fields:
        assert f"built.{field} == live.{field}" in exact
    assert (
        "built.storage_owner.unsafeGetStorageImpl() == built.storage_impl"
        in exact
    )

    end_preflight = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::verify_kernel_register_end_preflight",
        )
    )
    retained_validation = end_preflight.index(
        "validate_kernel_register_retained_occurrences()"
    )
    outer_fast_preflight = end_preflight.index(
        "verify_kernel_register_outer_fast_preflight()"
    )
    assert retained_validation < outer_fast_preflight

    retained = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::validate_kernel_register_retained_occurrences",
        )
    )
    owner_check = retained.index(
        "operand.storage_owner.unsafeGetStorageImpl()"
    )
    tensor_owner = retained.index(
        "const at::Tensor& tensor = operand.tensor_owner"
    )
    register_snapshot = retained.index(
        "query_register_state_token(", owner_check
    )
    assert tensor_owner < owner_check < register_snapshot
    assert "read_kernel_reg(" not in retained


def test_get_kernel_binds_the_exact_clone_before_reset_writer_and_enqueue():
    get_anchors = (
        (
            "::rhino_lkn::Kernel_t* "
            "RpuKernelGraph::get_kernel(KernelId id)",
            "admit_kernel_register_lookup(id)",
        ),
        (
            "::rhino_lkn::Kernel_t* "
            "RpuKernelGraph::get_kernel(const std::string& name)",
            "admit_kernel_register_lookup(name)",
        ),
    )
    for anchor, admission_token in get_anchors:
        body = _squash(
            _cpp_block("src/graph/graph_runtime_execute.cpp", anchor)
        )
        admission = body.index(admission_token)
        dispatch = body.index("switch (state_)")
        assert admission < dispatch
        for lookup_or_clone in (
            "KernelCache::instance()",
            "KernelCache::instance().get_program(",
            "std::make_unique<::rhino_lkn::Kernel_t>(",
        ):
            assert lookup_or_clone not in body[:admission]

    reset_anchors = (
        "::rhino_lkn::Kernel_t* "
        "RpuKernelGraph::get_kernel_reset(KernelId id)",
        "::rhino_lkn::Kernel_t* "
        "RpuKernelGraph::get_kernel_reset(const std::string& name)",
    )
    for anchor in reset_anchors:
        body = _squash(
            _cpp_block("src/graph/graph_runtime_execute.cpp", anchor)
        )
        get = body.index("get_kernel(")
        bind = body.index("bind_pending_kernel_register_lookup(*k)")
        reset = body.index("k->reset_regs()")
        returned = body.index("return k")
        assert get < bind < reset < returned
        assert body.count("bind_pending_kernel_register_lookup(") == 1

    binding = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::bind_pending_kernel_register_lookup",
        )
    )
    recording_pointer = binding.index(
        "kernels_[pending_kernel_idx_].get() == &kernel"
    )
    replay_pointer = binding.index(
        "kernels_[node.kernel_idx].get() == &kernel"
    )
    store_pointer = binding.index("pending.lookup_kernel = &kernel")
    assert recording_pointer < replay_pointer < store_pointer

    writer = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::write_pending_kernel_ddr_registers",
        )
    )
    assert "pending.lookup_kernel == &kernel" in writer
    assert "pending.written_kernel = &kernel" in writer
    consume = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::consume_pending_kernel_register_census",
        )
    )
    assert "pending.lookup_kernel == &kernel" in consume
    assert "pending.written_kernel == &kernel" in consume

    enqueue = _squash(
        _cpp_block(
            "src/graph/graph_runtime_execute.cpp",
            "void RpuKernelGraph::enqueue(::rhino_lkn::Kernel_t& kernel,\n"
            "                             const std::vector<uint16_t>& grid_dims,\n"
            "                             const std::vector<uint8_t>& core_ids,\n"
            "                             const QueueLaunchState& queue_state)",
        )
    )
    recording_consume = enqueue.index(
        "consume_pending_kernel_register_census("
    )
    node_append = enqueue.index("nodes_.push_back(", recording_consume)
    replay_consume = enqueue.index(
        "consume_pending_kernel_register_census(", node_append
    )
    replay_cursor = enqueue.index("cursor_++", replay_consume)
    assert recording_consume < node_append < replay_consume < replay_cursor

    replay_branch, _ = _block_after(enqueue, "case State::REPLAYING:")
    replay_consume = replay_branch.rindex(
        "consume_pending_kernel_register_census("
    )
    grid_check = replay_branch.rfind(
        "grid_dims == kn.grid_dims", 0, replay_consume
    )
    core_check = replay_branch.rfind(
        "core_ids == kn.core_ids", 0, replay_consume
    )
    queue_check = replay_branch.rfind(
        "queue_state == kn.queue_state", 0, replay_consume
    )
    replay_cursor = replay_branch.index("cursor_++", replay_consume)
    assert 0 <= grid_check < core_check < queue_check < replay_consume
    assert replay_consume < replay_cursor

    end = _squash(
        _function_body("src/graph/graph_runtime.cpp", "RpuKernelGraph::end")
    )
    preflight = end.index("verify_kernel_register_end_preflight()")
    first_flush = end.index("flush_boundary_ptrs()")
    first_execution = min(
        position
        for token in (
            "execute_graph_for_recording()",
            "execute_graph_oneshot()",
            "execute_graph_for_replaying()",
        )
        if (position := end.find(token)) >= 0
    )
    assert preflight < first_flush < first_execution


def test_gr00t_z2_facade_and_runtime_own_exactly_one_full_graph_scope():
    source = _function_body(
        "src/fused/rpu_gr00t_spm_z2.cpp",
        "rpu_gr00t_spm_z2_pipeline_forward",
    )
    guard = source.index("GraphKernelRegisterCensusGuard register_census(")
    producer = source.index("rpu_gr00t_vl_encoder_forward_z2(")
    checkpoints = _positions(source, r"\bregister_census\.checkpoint\s*\(")
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    consumer = source.index("forward_denoise_prepared_masks(")
    physical_postflight = source.index(
        "validate_active_pipeline_locked(", consumer
    )
    complete = source.index("register_census.complete()")
    final_return = source.index("return ", complete)

    assert (
        guard
        < producer
        < checkpoint
        < consumer
        < physical_postflight
        < complete
        < final_return
    )
    assert "epoch" in source[guard:producer]
    assert "plan_hash" in source[guard:producer]
    fast_commit = source.index(
        "register_census.commit_outer_fast_replay(", guard
    )
    fast_return = source.index("return output;", fast_commit)
    assert guard < fast_commit < fast_return < producer
    assert "return " not in source[producer:complete]
    assert source.count("return output;") == 2

    begin = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::begin_kernel_register_census",
        )
    )
    assert begin.count("active_stack_.size() == 1") == 1
    depth = begin.index("active_stack_.size() == 1")
    top = begin.index("active_stack_.back() == this")
    state = begin.index("state_ == State::RECORDING")
    assert depth < top < state


def test_admitted_launchers_have_no_raw_kernel_or_queue_escape():
    admitted = {
        "fp16 linear": _fp16_linear_body(),
        "K-cache insert": _kv_launch_once_body("k"),
        "V-cache insert": _kv_launch_once_body("v"),
        "SPM SDPA": _sdpa_body(),
        "ring allreduce": _allreduce_ring_body(),
    }
    for label, source in admitted.items():
        assert "QueueCache::instance" not in source, label
        assert "::rhino_lkn::Queue_t" not in source, label
        assert "std::make_unique<Queue_t" not in source, label
        assert "new Queue_t" not in source, label
        assert "GET_QUEUE(" in source, label

        raw_fallbacks = _positions(
            source, r"\bKernelCache::instance\(\)\.get_kernel\s*\("
        )
        if label not in {"K-cache insert", "V-cache insert"}:
            assert not raw_fallbacks, label
            continue

        # The legacy string lookup remains a PASSTHROUGH compatibility fallback.
        # During an active Graph, the preceding enum GET either succeeds or throws,
        # so the census-covered route cannot reach this raw lookup.
        assert len(raw_fallbacks) == 1, label
        fallback = raw_fallbacks[0]
        graph_get = source.index("GET_KERNEL(")
        guard = source.rfind("if (kernel == nullptr && !use_v16)", 0, fallback)
        assert 0 <= graph_get < guard < fallback, label

    facade = _function_body(
        "src/fused/rpu_gr00t_spm_z2.cpp",
        "rpu_gr00t_spm_z2_pipeline_forward",
    )
    for token in (
        "KernelCache::instance",
        "QueueCache::instance",
        "Queue_t",
        "enqueu_kernel",
    ):
        assert token not in facade


def test_dynamic_no_ddr_exception_is_exact_not_prefix_based():
    graph_sources = "\n".join(
        _read(relative) for relative in GRAPH_RUNTIME_SOURCES
    )
    exact = re.compile(
        r"(?:kernel_name|name)\s*==\s*\"binary_1xC_NxC_tileC\""
    )
    assert len(exact.findall(graph_sources)) == 1
    assert '"binary_1xC_NxC_tileC_bopa"' not in graph_sources
    assert "starts_with(\"binary_1xC_NxC_tileC" not in graph_sources
    assert "rfind(\"binary_1xC_NxC_tileC" not in graph_sources


def test_raw_direct_and_fast_replay_escape_paths_fail_closed():
    enqueue = _squash(
        _cpp_block(
            "src/graph/graph_runtime_execute.cpp",
            "void RpuKernelGraph::enqueue(::rhino_lkn::Kernel_t& kernel,\n"
            "                             const std::vector<uint16_t>& grid_dims,\n"
            "                             const std::vector<uint8_t>& core_ids,\n"
            "                             const QueueLaunchState& queue_state)",
        )
    )
    _, raw_gate_end = _census_escape_gate(enqueue)
    raw_flush = enqueue.index("flush_boundary_ptrs()", raw_gate_end)
    raw_oneshot = enqueue.index("execute_graph_oneshot()", raw_flush)
    raw_downgrade = enqueue.index("replayable_ = false", raw_oneshot)
    assert raw_gate_end < raw_flush < raw_oneshot < raw_downgrade

    sync = _squash(
        _function_body(
            "src/graph/graph_runtime_execute.cpp", "RpuKernelGraph::sync_point"
        )
    )
    _, sync_gate_end = _census_escape_gate(sync)
    assert sync_gate_end < sync.index("flush_boundary_ptrs()")

    mark_non_replayable = _squash(
        _function_body(
            "src/graph/graph_runtime.cpp",
            "RpuKernelGraph::mark_non_replayable",
        )
    )
    _, mark_gate_end = _census_escape_gate(mark_non_replayable)
    assert mark_gate_end < mark_non_replayable.index("replayable_ = false")

    fast_replay_escapes = {
        "RpuKernelGraph::skip_op_stream_for_fast_replay": (
            "cursor_ = nodes_.size()"
        ),
        "RpuKernelGraph::mark_post_fn_cursor": (
            "if (state_ == State::RECORDING)"
        ),
        "RpuKernelGraph::skip_layer_body_for_fast_replay": (
            "cursor_ = post_fn_cursor_"
        ),
        "RpuKernelGraph::skip_op_stream_with_patches": (
            "for (const auto& p : register_patches)"
        ),
    }
    for function, first_escape_side_effect in fast_replay_escapes.items():
        body = _squash(
            _function_body(
                "src/graph/graph_runtime_execute.cpp", function
            )
        )
        _, gate_end = _census_escape_gate(body)
        assert gate_end < body.index(first_escape_side_effect)

    direct = _squash(
        _function_body(
            "src/graph/graph_runtime.cpp",
            "RpuKernelGraph::check_direct_immediate_dma_allowed",
        )
    )
    assert "active_stack_.size() == 1" in direct
    direct_states = (
        "!graph->kernel_register_census_active_",
        "!graph->has_kernel_register_census_nodes_",
        "!graph->kernel_register_census_complete_",
        "!graph->kernel_register_census_poisoned_",
        "!graph->kernel_register_census_invocation_armed_",
        "!graph->pending_kernel_register_census_.has_value()",
    )
    direct_gate = direct.index("const bool plain_downgraded_passthrough")
    admission = direct.index("TORCH_CHECK( plain_downgraded_passthrough")
    for state in direct_states:
        assert direct_gate < direct.index(state) < admission


def test_child_graph_and_patch_escapes_are_rejected_by_the_census():
    child_escapes = (
        "RpuKernelGraph::register_child",
        "RpuKernelGraph::patch_child_inputs",
        "RpuKernelGraph::patch_child_data",
        "RpuKernelGraph::enter_replay_child_skip",
        "RpuKernelGraph::record_branch",
        "RpuKernelGraph::replay_prepared_child",
        "RpuKernelGraph::replay_prepared_child_with_data_patches",
        "RpuKernelGraph::extract_child_window",
        "RpuKernelGraph::replace_window_with_child",
    )
    for function in child_escapes:
        body = _squash(
            _function_body("src/graph/graph_runtime_child.cpp", function)
        )
        gate_start, _ = _census_escape_gate(body)
        if function == "RpuKernelGraph::extract_child_window":
            assert body.index("#if !RHINO_LAUNCH_HAS_DEV_PROGRAM_API") < gate_start
            assert "TORCH_CHECK( false" in body[:gate_start]
        else:
            assert gate_start == body.index("TORCH_CHECK")


def test_abort_end_and_invalidate_clear_invocation_and_build_authority():
    invocation_clear = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::clear_kernel_register_invocation_state",
        )
    )
    for reset in (
        "pending_kernel_register_census_.reset()",
        "kernel_register_census_active_ = false",
        "kernel_register_census_complete_ = false",
        "kernel_register_census_poisoned_ = false",
        "kernel_register_census_invocation_armed_ = false",
        "kernel_register_census_live_epoch_ = 0",
        "kernel_register_census_node_begin_ = 0",
        "kernel_register_census_next_phase_ = 0",
        "next_kernel_register_writer_ticket_ = 0",
        "kernel_register_census_stats_ = {}",
        "kernel_register_census_phase_begin_stats_ = {}",
        "kernel_register_census_rule_counts_.clear()",
        "kernel_register_outer_fast_force_post_visibility_owners_.clear()",
        "kernel_register_outer_fast_invocation_force_post_visibility_.clear()",
    ):
        assert invocation_clear.count(reset) == 1

    graph_implementation = " ".join(
        _squash(_read(source))
        for source in GRAPH_RUNTIME_SOURCES
        if source.endswith(".cpp")
    )
    assert graph_implementation.count(
        "kernel_register_census_poisoned_ = false"
    ) == 1
    assert graph_implementation.count(
        "clear_kernel_register_invocation_state();"
    ) == 3
    assert graph_implementation.count(
        "clear_kernel_register_build_state();"
    ) == 2
    cancel = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::cancel_kernel_register_census",
        )
    )
    assert cancel.count("kernel_register_census_poisoned_ = true") == 1
    assert "clear_kernel_register_invocation_state()" not in cancel
    assert "clear_kernel_register_build_state()" not in cancel
    assert cancel.index("kernel_register_census_active_ = false") < cancel.index(
        "kernel_register_census_poisoned_ = true"
    )

    build_clear = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::clear_kernel_register_build_state",
        )
    )
    invocation = build_clear.index("clear_kernel_register_invocation_state()")
    for reset in (
        "has_kernel_register_census_nodes_ = false",
        "kernel_register_census_build_committed_ = false",
        "kernel_register_census_plan_hash_ = 0",
        "kernel_register_census_policy_digest_ = 0",
        "kernel_register_census_policy_ = {}",
    ):
        assert invocation < build_clear.index(reset)

    begin_impl = _squash(
        _function_body("src/graph/graph_runtime.cpp", "RpuKernelGraph::begin_impl")
    )
    fresh_branch = begin_impl.index("if (state_ == State::PASSTHROUGH)")
    fresh_clear = begin_impl.index(
        "clear_kernel_register_build_state()", fresh_branch
    )
    replay_branch = begin_impl.index("else if (hit_built)", fresh_clear)
    replay_clear = begin_impl.index(
        "clear_kernel_register_invocation_state()", replay_branch
    )
    state_switch = begin_impl.index("switch (state_)", replay_clear)
    assert fresh_branch < fresh_clear < replay_branch < replay_clear < state_switch

    invalidate = _squash(
        _function_body(
            "src/graph/graph_runtime.cpp", "RpuKernelGraph::invalidate_impl"
        )
    )
    clear_build = invalidate.index("clear_kernel_register_build_state()")
    passthrough = invalidate.index("state_ = State::PASSTHROUGH")
    assert clear_build < passthrough

    abort = _squash(
        _function_body("src/graph/graph_runtime.cpp", "RpuKernelGraph::abort")
    )
    abort_clear = abort.index("invalidate_impl(")
    abort_pop = abort.index("active_stack_.pop_back()")
    assert abort_clear < abort_pop

    end = _squash(
        _function_body("src/graph/graph_runtime.cpp", "RpuKernelGraph::end")
    )
    stats_snapshot = end.index("last_stats_.register_census =")
    invocation_clear = end.index("clear_kernel_register_invocation_state()")
    active_pop = end.index("active_stack_.pop_back()")
    assert stats_snapshot < invocation_clear < active_pop


def test_schema5_outer_fast_uses_the_retained_catalog_without_fast_node_walks():
    fast = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::validate_kernel_register_outer_fast_replay",
        )
    )
    assert re.search(r"\bnodes_\s*(?:\[|\.size\s*\()", fast) is None
    assert fast.count("validate_kernel_register_retained_occurrences()") == 1

    commit = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::commit_kernel_register_outer_fast_replay",
        )
    )
    assert "validate_kernel_register_retained_occurrences" not in commit
    assert commit.index("commit_kernel_register_outer_fast_prepared(true)") < (
        commit.index("ticket.release()")
    )

    complete = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::complete_kernel_register_census",
        )
    )
    assert complete.index("build_kernel_register_retained_catalog()") < (
        complete.index("complete_kernel_register_outer_fast_ordinary()")
    )

    build_catalog = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::build_kernel_register_retained_catalog",
        )
    )
    assert "state_ == State::RECORDING" in build_catalog
    assert "kernel_register_retained_catalog_.empty()" in build_catalog
    assert (
        "for (size_t node_index = 0; node_index < nodes_.size();"
        in build_catalog
    )
    assert "kernel_register_retained_catalog_.swap(catalog)" in build_catalog

    validate_body = _function_body(
        "src/graph/graph_register_census.cpp",
        "RpuKernelGraph::validate_kernel_register_retained_occurrences",
    )
    validate_catalog = _squash(validate_body)
    assert "for (const auto& ref : kernel_register_retained_catalog_)" in (
        validate_catalog
    )
    assert "for (const auto& node : nodes_)" not in validate_catalog

    owner_snapshot = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "check_same_retained_owner_snapshot",
        )
    )
    for field in (
        "tensor_impl",
        "storage_impl",
        "data_ptr",
        "storage_data_ptr",
        "dtype",
        "device",
        "layout",
        "sizes",
        "strides",
        "storage_offset",
        "tensor_nbytes",
        "storage_nbytes",
        "dev_addr",
        "tensor_topology_id",
    ):
        assert f"first.{field} == occurrence.{field}" in owner_snapshot
    assert "first.abi == occurrence.abi" not in owner_snapshot
    assert (
        "occurrence.tensor_owner.unsafeGetTensorImpl() == "
        "occurrence.tensor_impl"
    ) in owner_snapshot
    assert (
        "occurrence.storage_owner.unsafeGetStorageImpl() == "
        "occurrence.storage_impl"
    ) in owner_snapshot

    owner_map = validate_catalog.index("validated_owners.reserve(")
    role_snapshot = validate_catalog.index(
        "validate_ddr_register_role_dtype( operand.abi.role, operand.dtype)"
    )
    role_live = validate_catalog.index(
        "validate_ddr_register_role_dtype( operand.abi.role, "
        "tensor.scalar_type())"
    )
    owner_insert = validate_catalog.index(
        "validated_owners.emplace(operand.tensor_impl, &operand)"
    )
    duplicate_snapshot = validate_catalog.index(
        "check_same_retained_owner_snapshot(*owner_it->second, operand)"
    )
    unique_live = validate_catalog.index("tensor_topology_id(")
    register_query = validate_catalog.index("query_register_state_token(")
    assert owner_map < role_snapshot < role_live < owner_insert
    assert owner_insert < duplicate_snapshot < register_query
    assert owner_insert < unique_live < register_query
    assert validate_catalog.count("RpuGetDevAddr(") == 1
    assert validate_catalog.count("query_register_state_token(") == 1
    assert "std::unordered_map<const c10::TensorImpl*," in validate_catalog
    assert "uint64_t live_state_token = 0" in validate_catalog
    assert "census.register_state_token != 0" in validate_catalog
    assert "live_state_token == census.register_state_token" in validate_catalog
    assert "read_kernel_reg(" not in validate_catalog
    assert (
        re.search(
            r"\b(?:static|thread_local)\b[^;]*validated_owners",
            validate_body,
            re.DOTALL,
        )
        is None
    )
    assert "validated_owners" not in _read("src/graph/graph_runtime.h")
    assert (
        "return kernel_register_retained_catalog_.size();"
        in validate_catalog
    )
    assert "return validated_owners.size();" not in validate_catalog

    duplicate_branch, duplicate_end = _block_after(
        validate_body, "if (!inserted)"
    )
    unique_brace = validate_body.index("{", duplicate_end)
    unique_end = _match_close(validate_body, unique_brace, "{", "}")
    unique_branch = validate_body[unique_brace + 1 : unique_end]
    for expensive in ("tensor_topology_id(", "RpuGetDevAddr("):
        assert expensive not in duplicate_branch
        assert expensive in unique_branch
    assert "query_register_state_token(" not in duplicate_branch
    assert "query_register_state_token(" not in unique_branch

    node_stale = validate_catalog.index("ref.node_index < nodes_.size()")
    kernel_stale = validate_catalog.index(
        "kernel_node.register_census.has_value()"
    )
    operand_stale = validate_catalog.index(
        "ref.operand_index < census.operands.size()"
    )
    assert node_stale < kernel_stale < operand_stale < owner_insert
    loop_body, _ = _block_after(
        validate_body,
        "for (const auto& ref : kernel_register_retained_catalog_)",
    )
    assert loop_body.count("query_register_state_token(") == 1

    launch_preflight = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::verify_kernel_register_outer_fast_launch_preflight",
        )
    )
    assert launch_preflight.count(
        "validate_kernel_register_retained_occurrences()"
    ) == 1

    clear = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::clear_kernel_register_build_state",
        )
    )
    assert clear.index("kernel_register_retained_catalog_.clear()") < clear.index(
        "kernel_register_outer_fast_build_state_ = {}"
    )


def test_schema5_build_seals_invocation_authority_owner_identity_and_order():
    build_authority = _squash(
        _cpp_block(
            "src/graph/graph_runtime.h",
            "struct OuterFastBuildInvocationAuthority",
        )
    )
    assert "std::weak_ptr<const uint64_t> prepared_owner" in build_authority
    assert "std::weak_ptr<uint64_t> consumed_owner" in build_authority

    build_state = _squash(
        _cpp_block("src/graph/graph_runtime.h", "struct OuterFastBuildState")
    )
    assert (
        "std::vector<OuterFastBuildInvocationAuthority> invocation_authorities"
        in build_state
    )
    assert "invocation_authority_count" not in build_state

    stage = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::stage_kernel_register_outer_fast_invocation_authority",
        )
    )
    distinct_pair = stage.index(
        "!shared_owner_aliases(prepared_owner, consumed_owner)"
    )
    duplicate_loop = stage.index(
        "for (const auto& current : candidate.invocation_authorities)"
    )
    publish = stage.index("candidate.invocation_authorities.push_back(")
    for pair in (
        "shared_owner_aliases(prepared_owner, current.prepared_owner)",
        "shared_owner_aliases(consumed_owner, current.consumed_owner)",
        "shared_owner_aliases(prepared_owner, current.consumed_owner)",
        "shared_owner_aliases(consumed_owner, current.prepared_owner)",
    ):
        assert pair in stage[duplicate_loop:publish]
    assert distinct_pair < duplicate_loop < publish

    ordinary = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::complete_kernel_register_outer_fast_ordinary",
        )
    )
    build_branch, _ = _block_after(ordinary, "if (state_ == State::RECORDING)")
    reserve = build_branch.index("build.invocation_authorities.reserve(")
    seal_loop = build_branch.index(
        "for (const auto& authority : candidate.invocation_authorities)"
    )
    seal = build_branch.index("OuterFastBuildInvocationAuthority", seal_loop)
    prepared = build_branch.index("authority.prepared_owner", seal)
    consumed = build_branch.index("authority.consumed_owner", prepared)
    assert reserve < seal_loop < seal < prepared < consumed
    assert "build.invocation_authorities.size()" in ordinary
    assert "invocation_authorities[i].prepared_owner.lock()" in ordinary
    assert "invocation_authorities[i].consumed_owner.lock()" in ordinary
    assert ordinary.count("same_shared_owner_identity(") >= 2

    fast = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::validate_kernel_register_outer_fast_replay",
        )
    )
    size_check = fast.index(
        "candidate.invocation_authorities.size() == "
        "build.invocation_authorities.size()"
    )
    indexed_check = fast.index(
        "for (size_t i = 0; i < build.invocation_authorities.size(); ++i)"
    )
    assert size_check < indexed_check
    assert "build.invocation_authorities[i].prepared_owner.lock()" in fast
    assert "build.invocation_authorities[i].consumed_owner.lock()" in fast
    assert fast.count("same_shared_owner_identity(") >= 2


def test_schema5_fast_visibility_is_minimal_and_ordinary_keeps_op_flushes():
    ordinary = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::complete_kernel_register_outer_fast_ordinary",
        )
    )
    assert "prepared.boundary_visibility = boundary_flush_ptrs_" in ordinary
    ordinary_visibility = ordinary[
        ordinary.index("prepared.boundary_visibility = boundary_flush_ptrs_") :
    ]
    ordinary_bursts, _ = _block_after(
        ordinary_visibility,
        "for (const auto& burst : candidate.ordinary_dma_bursts)",
    )
    ordinary_mutable_source, _ = _block_after(
        ordinary_bursts,
        "if (burst.variant == DmaNodeData::Variant::MutableSrc)",
    )
    assert "force_post_visibility" not in ordinary_mutable_source
    ordinary_destination, _ = _block_after(
        ordinary_bursts,
        "if (burst.side == GraphOuterFastDmaSide::Destination)",
    )
    for visibility in (
        "prepared.force_visibility_owners.push_back(burst.owner)",
        "prepared.force_visibility.push_back(",
        "prepared.force_post_visibility_owners.push_back(burst.owner)",
        "prepared.force_post_visibility.push_back(",
    ):
        assert visibility in ordinary_destination
    assert ordinary_visibility.count(
        "prepared.force_post_visibility.push_back("
    ) == 1
    assert ordinary.index(
        "sort_unique_ptrs(prepared.force_post_visibility)"
    ) < ordinary.index(
        "kernel_register_outer_fast_prepared_.emplace(std::move(prepared))"
    )

    slot_publish = ordinary.index(
        "*burst.live_base == burst.owner.dev_addr"
    )
    slot_prepare = ordinary.index("prepared.slot_writes.emplace_back(")
    assert slot_publish < slot_prepare

    commit = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::commit_kernel_register_outer_fast_prepared",
        )
    )
    fast_only, _ = _block_after(commit, "if (fast)")
    assert "*slot = value" in fast_only
    assert commit.count("*slot = value") == 1

    fast = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::validate_kernel_register_outer_fast_replay",
        )
    )
    assert "prepared.boundary_visibility = boundary_flush_ptrs_" not in fast
    assert fast.count("prepared.boundary_visibility.push_back(") == 3
    assert fast.count("prepared.force_visibility.push_back(") == 3
    assert fast.count("prepared.force_post_visibility.push_back(") == 2

    fixed_owners, _ = _block_after(
        fast, "for (const auto& owner : build.fixed_dma_owners)"
    )
    fixed_destination, _ = _block_after(
        fixed_owners,
        "if (owner.side == GraphOuterFastDmaSide::Destination)",
    )
    assert "prepared.normal_visibility_owners.push_back(owner.owner)" in (
        fixed_destination
    )
    assert "prepared.boundary_visibility.push_back(" in fixed_destination
    assert "prepared.force_visibility_owners.push_back(owner.owner)" in (
        fixed_destination
    )
    assert "prepared.force_visibility.push_back(" in fixed_destination
    assert "prepared.force_post_visibility_owners.push_back(owner.owner)" in (
        fixed_destination
    )
    assert "prepared.force_post_visibility.push_back(" in fixed_destination
    assert "GraphOuterFastDmaSide::Source" not in fixed_owners

    fixed_publish = fast.index(
        "prepared.force_post_visibility_owners.push_back(owner.owner)"
    )
    force_post_dedup = fast.index(
        "sort_unique_ptrs(prepared.force_post_visibility)"
    )
    ticket_publish = fast.index(
        "kernel_register_outer_fast_prepared_.emplace(std::move(prepared))"
    )
    assert fixed_publish < force_post_dedup < ticket_publish

    explicit_fixed, _ = _block_after(
        fast, "for (const auto& current : candidate.invocation_visibility)"
    )
    assert "check_kernel_register_outer_fast_tensor_exact(" in explicit_fixed
    assert (
        "prepared.normal_visibility_owners.push_back(current)"
        in explicit_fixed
    )
    assert "prepared.boundary_visibility.push_back(" in explicit_fixed

    mutable, _ = _block_after(
        fast, "for (const auto& binding : candidate.fast_mutable_dma_bindings)"
    )
    source, _ = _block_after(
        mutable, "if (slot.side == GraphOuterFastDmaSide::Source)"
    )
    destination, _ = _block_after(
        mutable, "if (slot.side == GraphOuterFastDmaSide::Destination)"
    )
    assert "prepared.force_visibility_owners.push_back(binding.owner)" in source
    assert "prepared.force_visibility.push_back(" in source
    assert "force_post_visibility" not in source
    assert "prepared.normal_visibility_owners.push_back(binding.owner)" in (
        destination
    )
    assert "prepared.boundary_visibility.push_back(" in destination
    assert "prepared.force_visibility_owners.push_back(binding.owner)" in (
        destination
    )
    assert "prepared.force_visibility.push_back(" in destination
    assert "prepared.force_post_visibility_owners.push_back(binding.owner)" in (
        destination
    )
    assert "prepared.force_post_visibility.push_back(" in destination

    force = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::flush_kernel_register_outer_fast_force_visibility",
        )
    )
    for authority in (
        "kernel_register_outer_fast_build_state_.staged",
        "kernel_register_census_invocation_armed_",
        "kernel_register_outer_fast_invocation_force_visibility_.empty()",
        "g_rpu_ddr_flush_force_enabled",
    ):
        assert authority in force
    assert "kernel_register_outer_fast_hit_" not in force

    force_post = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::flush_kernel_register_outer_fast_force_post_visibility",
        )
    )
    for authority in (
        "kernel_register_outer_fast_build_state_.staged",
        "kernel_register_census_invocation_armed_",
        "kernel_register_outer_fast_invocation_force_post_visibility_.empty()",
        "g_rpu_ddr_flush_force_enabled",
    ):
        assert authority in force_post
    # Normal-off force publication is exercised below with the actual methods;
    # normal-on may reuse a publication from the immediately preceding pass.
    assert "kernel_register_outer_fast_hit_" not in force_post
    assert "kernel_register_outer_fast_invocation_force_visibility_" not in (
        force_post
    )
    assert force_post.count("rhino_lkn::RpuDdrFlush(ptr)") == 1

    header = _squash(_read("src/graph/graph_runtime.h"))
    prepared = _squash(
        _cpp_block("src/graph/graph_runtime.h", "struct OuterFastPreparedCommit")
    )
    assert "std::vector<void*> force_post_visibility" in prepared
    assert (
        "std::vector<OuterFastTensorSnapshot> force_post_visibility_owners"
        in prepared
    )
    assert (
        "kernel_register_outer_fast_force_post_visibility_owners_" in header
    )
    assert (
        "kernel_register_outer_fast_invocation_force_post_visibility_" in header
    )

    commit = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::commit_kernel_register_outer_fast_prepared",
        )
    )
    ptr_swap = commit.index(
        "kernel_register_outer_fast_invocation_force_post_visibility_.swap("
    )
    owner_swap = commit.index(
        "kernel_register_outer_fast_force_post_visibility_owners_.swap("
    )
    publish = commit.index("kernel_register_outer_fast_hit_ = fast")
    assert ptr_swap < publish and owner_swap < publish


def test_gr00t_outer_fast_branch_skips_both_model_op_walks_atomically():
    body = _squash(
        _function_body(
            "src/fused/rpu_gr00t_spm_z2.cpp",
            "rpu_gr00t_spm_z2_pipeline_forward",
        )
    )
    fast, fast_end = _block_after(
        body, "if (RpuKernelGraph::active().state() =="
    )
    bind = fast.index("bind_vl_outer_fast_input(")
    prepare = fast.index("prepare_denoise_outer_fast_replay(")
    validate = fast.index("validate_outer_fast_replay()")
    postflight = fast.index("validate_active_pipeline_locked(", validate)
    digest = fast.index("physical_binding_digest(", postflight)
    commit = fast.index("commit_outer_fast_replay(", digest)
    returned = fast.index("return output", commit)
    assert bind < prepare < validate < postflight < digest < commit < returned

    for ordinary_only in (
        "rpu_gr00t_vl_encoder_forward_z2(",
        "forward_denoise_prepared_masks(",
        "register_census.checkpoint(",
        "register_census.complete(",
    ):
        assert ordinary_only not in fast
        assert ordinary_only in body[fast_end:]

    after_commit = fast[fast.index(";", commit) + 1 :]
    assert after_commit.count("return output") == 1
    for throwing_or_staging_work in (
        "TORCH_CHECK(",
        "validate_",
        "physical_binding_digest(",
        "prepare_",
        "bind_",
        "stage_",
    ):
        assert throwing_or_staging_work not in after_commit


def test_gr00t_mask_publication_is_fail_closed_and_graph_success_consumed():
    prepare = _squash(
        _function_body("src/fused/rpu_gr00t_dit_model.cpp", "prepare_z2_masks")
    )
    reject_overwrite = prepare.index("z2_masks_prepared_generation_ <=")
    poison = prepare.index("z2_masks_prepared_ = false")
    clear_plan = prepare.index("z2_masks_plan_hash_ = 0", poison)
    text_copy = prepare.index("prep_mask_into(tmask")
    image_copy = prepare.index("prep_mask_into(imask")
    bump = prepare.index("++*z2_mask_generation_")
    publish_generation = prepare.index(
        "z2_masks_prepared_generation_ = *z2_mask_generation_"
    )
    publish_plan = prepare.index("z2_masks_plan_hash_ = plan_hash")
    publish_ready = prepare.rindex("z2_masks_prepared_ = true")
    assert prepare.count("prep_mask_into(") == 2
    assert (
        reject_overwrite
        < poison
        < clear_plan
        < text_copy
        < image_copy
        < bump
        < publish_generation
        < publish_plan
        < publish_ready
    )
    assert "*z2_masks_consumed_generation_ =" not in prepare

    stage = _squash(
        _function_body(
            "src/fused/rpu_gr00t_dit_model.cpp",
            "stage_z2_outer_fast_invocation_authority",
        )
    )
    authority = stage.index("guard.stage_outer_fast_invocation_authority(")
    authority_end = _match_close(stage, stage.index("(", authority), "(", ")")
    call = stage[authority:authority_end]
    parameters = (
        "z2_mask_generation_",
        "z2_masks_consumed_generation_",
        "z2_masks_prepared_generation_",
    )
    assert all(call.count(parameter) == 1 for parameter in parameters)
    positions = [call.index(parameter) for parameter in parameters]
    assert positions == sorted(positions)
    visibility = _positions(
        stage, r"\bguard\.stage_outer_fast_invocation_visibility\s*\("
    )
    assert len(visibility) == 2
    assert authority < visibility[0] < visibility[1]

    ordinary = _squash(
        _function_body("src/fused/rpu_gr00t_dit_model.cpp", "unroll_forward_z2")
    )
    assert "*z2_masks_consumed_generation_ =" not in ordinary
    assert "z2_masks_prepared_ = false" not in ordinary

    mark = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::mark_kernel_register_outer_fast_execution_success",
        )
    )
    assert mark.count(
        "*authority.consumed_owner = authority.expected_generation"
    ) == 1
    end = _squash(
        _function_body("src/graph/graph_runtime.cpp", "RpuKernelGraph::end")
    )
    marks = _positions(
        end, r"\bmark_kernel_register_outer_fast_execution_success\s*\("
    )
    assert len(marks) == 2
    force_pre = _positions(
        end, r"\bflush_kernel_register_outer_fast_force_visibility\s*\("
    )
    force_post = _positions(
        end,
        r"\bflush_kernel_register_outer_fast_force_post_visibility\s*\(",
    )
    assert len(force_pre) == 2
    assert len(force_post) == 2
    build_execute = end.index("execute_graph_for_recording()")
    replay_execute = end.index("execute_graph_for_replaying()")
    assert force_pre[0] < build_execute < force_post[0] < marks[0]
    assert force_pre[1] < replay_execute < force_post[1] < marks[1]


def test_outer_fast_visibility_reuses_only_completed_same_pass_flushes(tmp_path):
    """Compile the real flush methods and BUILD/REPLAY execution windows.

    A fake cache tracks CPU-dirty inputs and device-dirty outputs.  Each pointer
    must be published once per applicable boundary, including with normal off;
    a PRE publication must never suppress the POST invalidation.
    """
    import shutil
    import subprocess

    compiler = shutil.which("g++")
    assert compiler is not None, "host contract requires a C++ compiler"
    relative = "src/graph/graph_register_census.cpp"
    methods = "\n".join(
        f"void RpuKernelGraph::{name}() {qualifier} {{\n"
        + _function_body_raw(path, f"RpuKernelGraph::{name}") + "\n}"
        for path, name, qualifier in (
            ("src/graph/graph_runtime.cpp", "flush_boundary_ptrs", ""),
            (relative, "flush_kernel_register_outer_fast_force_visibility", "const noexcept"),
            (relative, "flush_kernel_register_outer_fast_force_post_visibility", "const noexcept"),
        )
    )
    end = _function_body_raw("src/graph/graph_runtime.cpp", "RpuKernelGraph::end")
    stripped = _strip_cpp_comments_and_literals(end)
    replay = stripped.index("case State::REPLAYING:")
    windows = []
    for start, stop in ((0, replay), (replay, len(end))):
        begin = stripped.index("flush_boundary_ptrs();", start, stop)
        if start == replay:
            # Include the RECORD_FUNCTION scope enclosing the first replay flush.
            begin = max(
                match.start() for match in re.compile(r"\{").finditer(stripped, start, begin)
                if _match_close(stripped, match.start(), "{", "}") > begin
            )
        finish = stripped.index(
            "mark_kernel_register_outer_fast_execution_success();", begin, stop
        ) + len("mark_kernel_register_outer_fast_execution_success();")
        window = end[begin:finish]
        calls = re.findall(
            r"\b(flush_boundary_ptrs|flush_kernel_register_outer_fast_force_visibility|"
            r"flush_kernel_register_outer_fast_force_post_visibility)\s*\(",
            _strip_cpp_comments_and_literals(window),
        )
        assert calls == [
            "flush_boundary_ptrs", "flush_kernel_register_outer_fast_force_visibility",
            "flush_boundary_ptrs", "flush_kernel_register_outer_fast_force_post_visibility",
        ]
        windows.append(window)
    # Any future force caller must establish this same-pass prerequisite too.
    for name in ("force_visibility", "force_post_visibility"):
        assert len(re.findall(
            rf"\bflush_kernel_register_outer_fast_{name}\s*\(",
            _strip_cpp_comments_and_literals(_read("src/graph/graph_runtime.cpp")),
        )) == 2

    harness = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <vector>
#define RECORD_FUNCTION(...) do {} while (0)
bool g_rpu_ddr_flush_enabled = false;
bool g_rpu_ddr_flush_force_enabled = true;
struct Cell { int host, device; bool cpu_dirty; };
std::array<Cell, 8> cells;
std::array<std::vector<void*>, 2> publications;
int phase = 0;
namespace rhino_lkn {
void RpuDdrFlush(void* ptr) {
    publications[phase].push_back(ptr);
    auto& cell = *static_cast<Cell*>(ptr);
    if (cell.cpu_dirty) cell.device = cell.host;
    else cell.host = cell.device;
    cell.cpu_dirty = false;
}
void RpuDdrFlush(void*, size_t) {
    assert(false && "untyped legacy pointers must not become sized ranges");
}
}
struct RpuKernelGraph {
    struct OuterFastTensorSnapshot {
        const void* tensor_impl = nullptr;
        const void* storage_impl = nullptr;
        const void* data_ptr = nullptr;
        const void* storage_data_ptr = nullptr;
        int64_t storage_offset = 0;
        size_t tensor_nbytes = 0, storage_nbytes = 0;
        std::vector<int64_t> sizes, strides;
    };
    struct { bool staged = true; } kernel_register_outer_fast_build_state_;
    bool kernel_register_census_invocation_armed_ = true;
    std::vector<void*> boundary_flush_ptrs_;
    std::vector<void*> kernel_register_outer_fast_invocation_force_visibility_;
    std::vector<void*> kernel_register_outer_fast_invocation_force_post_visibility_;
    // Legacy untyped pointers have no range provenance and retain unsized flushes.
    std::vector<OuterFastTensorSnapshot> kernel_register_outer_fast_force_visibility_owners_;
    std::vector<OuterFastTensorSnapshot> kernel_register_outer_fast_force_post_visibility_owners_;
    struct { size_t boundary_flush_count = 0; } last_stats_;
    std::vector<int> segments_;
    std::set<void*> expected_pre, expected_post;
    int executions = 0, marks = 0, preflights = 0;
    bool oneshot = false;
    void flush_boundary_ptrs();
    void flush_kernel_register_outer_fast_force_visibility() const noexcept;
    void flush_kernel_register_outer_fast_force_post_visibility() const noexcept;
    void verify_kernel_register_outer_fast_launch_preflight() { ++preflights; }
    bool force_oneshot_on_replay_enabled() { return oneshot; }
    void build_segments_from_nodes() {}
    void execute_graph_for_recording() { execute(); }
    void execute_graph_for_replaying() { execute(); }
    void execute_graph_oneshot() { execute(); }
    void execute() {
        assert(phase == 0 && preflights == 1 && executions++ == 0);
        for (size_t i = 0; i < cells.size(); ++i) {
            assert(cells[i].device == (expected_pre.count(&cells[i]) ? int(i + 10) : -1));
            cells[i].device = int(i + 1000);
            cells[i].cpu_dirty = false;
        }
        phase = 1;
    }
    void mark_kernel_register_outer_fast_execution_success() {
        assert(phase == 1 && executions == 1 && marks++ == 0);
        for (size_t i = 0; i < cells.size(); ++i)
            assert(cells[i].host == (expected_post.count(&cells[i]) ? int(i + 1000) : int(i + 10)));
    }
    void build();
    void replay();
};
'''
    main = r'''
int main() {
    // Partial/disjoint/full overlap, no normal entries, no force entries, and
    // the measured LingBot2 four-normal/eight-pre/four-post shape.
    const std::vector<std::array<std::vector<int>, 3>> shapes = {
        {{{3, 1, 3}, {0, 1, 2}, {1, 4}}},
        {{{3, 1, 3}, {0, 2}, {4, 5}}},
        {{{3, 1, 3}, {1, 3}, {1, 3}}},
        {{{}, {0, 1}, {1, 4}}},
        {{{3, 1, 3}, {}, {}}},
        {{{7, 5, 3, 1, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {1, 3, 5, 7}}},
    };
    int cases = 0;
    for (const auto& shape : shapes)
    for (bool normal : {false, true})
    for (bool force : {false, true})
    for (bool staged : {false, true})
    for (bool armed : {false, true})
    for (int mode : {0, 1, 2}) {
        g_rpu_ddr_flush_enabled = normal;
        g_rpu_ddr_flush_force_enabled = force;
        phase = 0;
        publications = {};
        for (size_t i = 0; i < cells.size(); ++i)
            cells[i] = {int(i + 10), -1, true};
        RpuKernelGraph graph;
        graph.kernel_register_outer_fast_build_state_.staged = staged;
        graph.kernel_register_census_invocation_armed_ = armed;
        graph.oneshot = mode == 2;
        for (int i : shape[0]) graph.boundary_flush_ptrs_.push_back(&cells[i]);
        for (int i : shape[1]) graph.kernel_register_outer_fast_invocation_force_visibility_.push_back(&cells[i]);
        for (int i : shape[2]) graph.kernel_register_outer_fast_invocation_force_post_visibility_.push_back(&cells[i]);
        if (normal) {
            graph.expected_pre.insert(graph.boundary_flush_ptrs_.begin(), graph.boundary_flush_ptrs_.end());
            graph.expected_post = graph.expected_pre;
        }
        if (force && staged && armed) {
            for (int i : shape[1]) graph.expected_pre.insert(&cells[i]);
            for (int i : shape[2]) graph.expected_post.insert(&cells[i]);
        }
        if (mode == 0) graph.build(); else graph.replay();
        assert(graph.executions == 1 && graph.marks == 1);
        for (int pass : {0, 1}) {
            const auto& expected = pass == 0 ? graph.expected_pre : graph.expected_post;
            const std::set<void*> actual(publications[pass].begin(), publications[pass].end());
            assert(actual == expected);
            assert(publications[pass].size() == expected.size());
        }
        ++cases;
    }
    assert(cases == 288);
    std::cout << cases << " visibility cases passed\n";
}
'''
    source = tmp_path / "outer_fast_visibility.cpp"
    source.write_text(
        harness + methods + "\nvoid RpuKernelGraph::build() {\n" + windows[0]
        + "\n}\nvoid RpuKernelGraph::replay() {\n" + windows[1] + "\n}\n" + main,
        encoding="utf-8",
    )
    binary = tmp_path / "outer_fast_visibility"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert run.stdout.strip() == "288 visibility cases passed"


def test_outer_fast_source_ranges_preserve_dirty_replay_and_owner_boundaries(tmp_path):
    """Run actual PRE/POST methods and end windows over separate byte caches.

    Snapshot fields are stand-ins; allocation, DMA/GEMM and the SDK cache API
    are simulated. The production span admission and publication code is real.
    """
    import shutil
    import subprocess

    compiler = shutil.which("g++")
    assert compiler is not None, "range publication regression requires C++"
    methods = "\n".join(
        f"void RpuKernelGraph::{name}() {qualifier} {{\n"
        + _function_body_raw(path, f"RpuKernelGraph::{name}") + "\n}"
        for path, name, qualifier in (
            ("src/graph/graph_runtime.cpp", "flush_boundary_ptrs", ""),
            ("src/graph/graph_register_census.cpp",
             "flush_kernel_register_outer_fast_force_visibility", "const noexcept"),
            ("src/graph/graph_register_census.cpp",
             "flush_kernel_register_outer_fast_force_post_visibility", "const noexcept"),
        )
    )
    record = _function_body_raw("src/graph/graph_runtime.cpp", "RpuKernelGraph::record_boundary_flush")
    end = _function_body_raw("src/graph/graph_runtime.cpp", "RpuKernelGraph::end")
    stripped = _strip_cpp_comments_and_literals(end)
    split = stripped.index("case State::REPLAYING:")
    windows = []
    for start, stop in ((0, split), (split, len(end))):
        begin = stripped.index("flush_boundary_ptrs();", start, stop)
        if start:
            begin = max(
                m.start() for m in re.finditer(r"\{", stripped[start:begin])
                if _match_close(stripped, start + m.start(), "{", "}") > begin
            ) + start
        finish = stripped.index("mark_kernel_register_outer_fast_execution_success();", begin, stop)
        windows.append(end[begin:finish + len("mark_kernel_register_outer_fast_execution_success();")])

    # HostCallback's actual schema-marked zero-copy writeback publication must
    # retain force semantics independently of the typed-source optimization.
    fallback = _read("src/graph/graph_cpu_fallback_stub.cpp")
    writeback_start = fallback.index("    for (void* ptr : rpu_backed_writes)")
    writeback = fallback[writeback_start:fallback.index("    }", writeback_start) + 5]
    profile = _read("src/core/rpu_profile.h")
    macro_start = profile.index("#define rpu_ddr_flush_force(")
    force_macro = profile[macro_start:].split("\n\n", 1)[0]
    harness = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define RECORD_FUNCTION(...) do {} while (0)
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
bool g_rpu_ddr_flush_enabled = false, g_rpu_ddr_flush_force_enabled = true;
struct Memory {
    std::vector<uint8_t> cpu, device, dirty;
    explicit Memory(size_t n=384):cpu(n,0),device(n,0),dirty(n,0) {}
    void write(size_t begin,size_t n,uint8_t value) {
        std::fill_n(cpu.begin()+begin,n,value);
        std::fill_n(dirty.begin()+begin,n,1);
    }
};
std::vector<Memory*> memories;
struct Publication { void* ptr;size_t bytes;bool sized; };
std::array<std::vector<Publication>,2> publications;
int phase=0;
namespace rhino_lkn {
void cache(void* ptr,size_t bytes,bool sized) {
    publications[phase].push_back({ptr,bytes,sized});
    for(auto* memory:memories) {
        auto base=reinterpret_cast<uintptr_t>(memory->cpu.data());
        auto address=reinterpret_cast<uintptr_t>(ptr);
        if(address<base || address-base>=memory->cpu.size()) continue;
        size_t begin=address-base,n=sized?bytes:memory->cpu.size()-begin;
        assert(n<=memory->cpu.size()-begin);
        for(size_t i=begin;i<begin+n;++i) {
            if(memory->dirty[i]) memory->device[i]=memory->cpu[i];
            else memory->cpu[i]=memory->device[i];
            memory->dirty[i]=0;
        }
        return;
    }
    assert(false && "SDK publication must use a live backing address");
}
void RpuDdrFlush(void* ptr) {cache(ptr,0,false);}
void RpuDdrFlush(void* ptr,size_t bytes) {cache(ptr,bytes,true);}
}
struct RpuKernelGraph {
    struct OuterFastTensorSnapshot {
        const void* tensor_impl=nullptr;
        const void* storage_impl=nullptr;
        const void* data_ptr=nullptr;
        const void* storage_data_ptr=nullptr;
        int64_t storage_offset=0;
        size_t tensor_nbytes=0,storage_nbytes=0;
        std::vector<int64_t> sizes,strides;
    };
    struct {bool staged=true;} kernel_register_outer_fast_build_state_;
    bool kernel_register_census_invocation_armed_=true;
    bool kernel_register_outer_fast_hit_=false;
    std::vector<void*> boundary_flush_ptrs_;
    std::vector<void*> kernel_register_outer_fast_invocation_force_visibility_;
    std::vector<void*> kernel_register_outer_fast_invocation_force_post_visibility_;
    std::vector<OuterFastTensorSnapshot> kernel_register_outer_fast_force_visibility_owners_;
    std::vector<OuterFastTensorSnapshot> kernel_register_outer_fast_force_post_visibility_owners_;
    struct {size_t boundary_flush_count=0;} last_stats_;
    std::vector<int> segments_;
    Memory *input=nullptr,*output=nullptr;
    uint8_t expected_input=0;
    int executions=0,marks=0,preflights=0;
    void flush_boundary_ptrs();
    void flush_kernel_register_outer_fast_force_visibility() const noexcept;
    void flush_kernel_register_outer_fast_force_post_visibility() const noexcept;
    void record_boundary_flush(void* ptr);
    void check_kernel_register_node_emission(const char*) {}
    void verify_kernel_register_outer_fast_launch_preflight() {++preflights;}
    bool force_oneshot_on_replay_enabled() {return false;}
    void build_segments_from_nodes() {}
    void execute_graph_for_recording() {execute();}
    void execute_graph_for_replaying() {execute();}
    void execute_graph_oneshot() {assert(false);}
    void execute() {
        assert(phase==0 && preflights==1 && executions++==0);
        for(size_t i=64;i<256;++i) assert(input->device[i]==expected_input);
        std::fill(output->device.begin(),output->device.end(),expected_input+1);
        std::fill(output->dirty.begin(),output->dirty.end(),0);
        phase=1;
    }
    void mark_kernel_register_outer_fast_execution_success() {
        assert(phase==1 && marks++==0 && executions==1);
        const uint8_t value=(g_rpu_ddr_flush_enabled||g_rpu_ddr_flush_force_enabled)?expected_input+1:0;
        for(auto byte:output->cpu) assert(byte==value);
    }
    void build();void replay();
};
using Snapshot=RpuKernelGraph::OuterFastTensorSnapshot;
Snapshot owner(Memory& memory,size_t begin,size_t bytes) {
    Snapshot s;
    s.tensor_impl=memory.cpu.data()+begin;
    s.storage_impl=&memory;s.data_ptr=memory.cpu.data()+begin;
    s.storage_data_ptr=memory.cpu.data();s.storage_offset=begin;
    s.tensor_nbytes=bytes;s.storage_nbytes=memory.cpu.size();
    s.sizes={static_cast<int64_t>(bytes)};s.strides={1};return s;
}
void sources(RpuKernelGraph& graph,Memory& memory,const std::vector<size_t>& offsets) {
    for(size_t offset:offsets) {
        auto s=owner(memory,offset,64);
        graph.kernel_register_outer_fast_force_visibility_owners_.push_back(s);
        graph.kernel_register_outer_fast_invocation_force_visibility_.push_back(const_cast<void*>(s.data_ptr));
    }
    std::sort(graph.kernel_register_outer_fast_invocation_force_visibility_.begin(),
              graph.kernel_register_outer_fast_invocation_force_visibility_.end());
    std::reverse(graph.kernel_register_outer_fast_force_visibility_owners_.begin(),
                 graph.kernel_register_outer_fast_force_visibility_owners_.end());
}
size_t sized_calls(int pass=0) {
    return std::count_if(publications[pass].begin(),publications[pass].end(),
                         [](const Publication& p){return p.sized;});
}
void reset() {phase=0;publications={};memories.clear();}
'''
    main = r'''
int main() {
    int geometry_cases=0,replay_cases=0,callback_cases=0;
    const std::vector<std::string> shapes={
        "adjacent3","adjacent2","ling_fp16","singleton_stride","exact_duplicate","distinct_storage","gap","overlap",
        "single","ambiguous","post_storage","backing_drift","noncontiguous","zero",
        "negative_shape","negative_stride","offset","out_of_storage","address_overflow",
        "size_overflow","normal_covered","staged_off","armed_off"
    };
    for(const auto& shape:shapes) {
        reset();Memory input(shape=="ling_fp16"?2097152:384);memories={&input};RpuKernelGraph graph;
        sources(graph,input,shape=="ling_fp16"?std::vector<size_t>{0,524288,1048576}:
                            shape=="gap"?std::vector<size_t>{64,192,256}:
                            shape=="overlap"?std::vector<size_t>{64,96,160}:
                            shape=="single"?std::vector<size_t>{64}:
                            shape=="adjacent2"?std::vector<size_t>{64,128}:std::vector<size_t>{64,128,192});
        auto& held=graph.kernel_register_outer_fast_force_visibility_owners_;
        if(shape=="ling_fp16") for(auto& s:held) {
            s.tensor_nbytes=524288;s.storage_offset/=2;
            s.sizes={1,256,1024};s.strides={262144,1024,1};
        }
        if(shape=="singleton_stride") for(auto& s:held) {s.sizes={1,64};s.strides={999,1};}
        if(shape=="exact_duplicate") held.push_back(held.front());
        if(shape=="distinct_storage") for(size_t i=0;i<held.size();++i)
            held[i].storage_impl=reinterpret_cast<void*>(i+1);
        if(shape=="ambiguous") {auto s=held.front();s.tensor_impl=&graph;held.push_back(s);}
        if(shape=="post_storage") graph.kernel_register_outer_fast_force_post_visibility_owners_.push_back(owner(input,320,64));
        if(shape=="backing_drift") held.front().storage_data_ptr=input.cpu.data()+1;
        if(shape=="noncontiguous") {held.front().sizes={2,32};held.front().strides={64,1};}
        if(shape=="zero") held.front().tensor_nbytes=0;
        if(shape=="negative_shape") held.front().sizes={-64};
        if(shape=="negative_stride") held.front().strides={-1};
        if(shape=="offset") held.front().storage_offset=-1;
        if(shape=="out_of_storage") held.front().tensor_nbytes=384;
        if(shape=="address_overflow") held.front().storage_nbytes=std::numeric_limits<size_t>::max();
        if(shape=="size_overflow") {held.front().sizes={INT64_MAX,3};held.front().strides={3,1};}
        if(shape=="normal_covered") graph.boundary_flush_ptrs_.push_back(input.cpu.data()+128);
        if(shape=="staged_off") graph.kernel_register_outer_fast_build_state_.staged=false;
        if(shape=="armed_off") graph.kernel_register_census_invocation_armed_=false;
        g_rpu_ddr_flush_enabled=shape=="normal_covered";g_rpu_ddr_flush_force_enabled=true;
        graph.flush_boundary_ptrs();graph.flush_kernel_register_outer_fast_force_visibility();
        const bool merged=shape=="adjacent3"||shape=="adjacent2"||shape=="exact_duplicate"||
                          shape=="ling_fp16"||shape=="singleton_stride";
        assert(sized_calls()==size_t(merged));
        if(merged) {
            assert(publications[0].size()==1 && publications[0][0].ptr==input.cpu.data()+(shape=="ling_fp16"?0:64));
            assert(publications[0][0].bytes==(shape=="ling_fp16"?1572864:shape=="adjacent2"?128:192));
        } else if(shape=="staged_off"||shape=="armed_off") assert(publications[0].empty());
        else assert(publications[0].size()==graph.kernel_register_outer_fast_invocation_force_visibility_.size());
        ++geometry_cases;
    }
    for(bool normal:{false,true}) for(bool force:{false,true}) {
        reset();Memory input;memories={&input};RpuKernelGraph graph;
        sources(graph,input,{64,128,192});graph.input=&input;
        const auto source_owners=graph.kernel_register_outer_fast_force_visibility_owners_;
        const auto source_ptrs=graph.kernel_register_outer_fast_invocation_force_visibility_;
        std::vector<Memory> output_owners;output_owners.reserve(5);
        std::vector<std::vector<uint8_t>> saved_outputs;
        const std::array<uint8_t,5> values={17,29,17,29,17};
        g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
        for(size_t leg=0;leg<values.size();++leg) {
            phase=0;publications={};graph.executions=graph.marks=graph.preflights=0;
            output_owners.emplace_back();auto& output=output_owners.back();
            memories.push_back(&output);graph.output=&output;
            const auto destination=owner(output,0,output.cpu.size());
            graph.kernel_register_outer_fast_force_visibility_owners_=source_owners;
            graph.kernel_register_outer_fast_invocation_force_visibility_=source_ptrs;
            graph.kernel_register_outer_fast_force_visibility_owners_.push_back(destination);
            graph.kernel_register_outer_fast_invocation_force_visibility_.push_back(output.cpu.data());
            std::sort(graph.kernel_register_outer_fast_invocation_force_visibility_.begin(),
                      graph.kernel_register_outer_fast_invocation_force_visibility_.end());
            graph.kernel_register_outer_fast_force_post_visibility_owners_={destination};
            graph.kernel_register_outer_fast_invocation_force_post_visibility_={output.cpu.data()};
            graph.boundary_flush_ptrs_={output.cpu.data()};
            graph.kernel_register_outer_fast_hit_=leg==2||leg==3;
            input.write(0,64,91);input.write(64,192,values[leg]);input.write(256,128,93);
            graph.expected_input=force?values[leg]:0;
            if(leg==0) graph.build();else graph.replay();
            assert(sized_calls()==size_t(force) && sized_calls(1)==0);
            for(size_t i=0;i<64;++i) assert(input.device[i]==0 && input.cpu[i]==91);
            for(size_t i=256;i<input.cpu.size();++i) assert(input.device[i]==0 && input.cpu[i]==93);
            if(force) {
                auto p=std::find_if(publications[0].begin(),publications[0].end(),[](const Publication& q){return q.sized;});
                assert(p->ptr==input.cpu.data()+64 && p->bytes==192);
                saved_outputs.push_back(output.cpu);
                if(leg==2||leg==4) assert(saved_outputs.back()==saved_outputs[0]);
                if(leg==1||leg==3) assert(saved_outputs.back()!=saved_outputs[0]);
                for(size_t i=0;i<saved_outputs.size();++i) assert(output_owners[i].cpu==saved_outputs[i]);
            }
            assert(normal==g_rpu_ddr_flush_enabled && force==g_rpu_ddr_flush_force_enabled);
            ++replay_cases;
        }
    }
    // HostCallback is outside typed census. Its actual fallback writeback loop
    // still publishes .out CPU writes with normal disabled and force enabled.
    for(bool normal:{false,true}) for(bool force:{false,true}) {
        reset();Memory output;memories={&output};RpuKernelGraph graph;
        graph.kernel_register_outer_fast_build_state_.staged=false;
        graph.kernel_register_census_invocation_armed_=false;
        g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
        output.write(0,output.cpu.size(),37);
        std::vector<void*> rpu_backed_writes={output.cpu.data()};
WRITEBACK
        graph.record_boundary_flush(output.cpu.data());
        graph.flush_boundary_ptrs();graph.flush_kernel_register_outer_fast_force_visibility();
        assert(sized_calls()==0);
        for(auto byte:output.device) assert(byte==(normal||force?37:0));
        assert(publications[0].size()==size_t(normal)+size_t(force));
        ++callback_cases;
    }
    std::cout<<geometry_cases<<" owner-range cases, "<<replay_cases
             <<" dirty replay cases, "<<callback_cases<<" callback flag cases passed\n";
}
'''.replace("WRITEBACK", writeback)
    source = tmp_path / "source_visibility_ranges.cpp"
    source.write_text(
        harness + force_macro + "\n" + methods
        + "\nvoid RpuKernelGraph::record_boundary_flush(void* ptr) {\n" + record + "\n}"
        + "\nvoid RpuKernelGraph::build() {\n" + windows[0] + "\n}"
        + "\nvoid RpuKernelGraph::replay() {\n" + windows[1] + "\n}\n" + main,
        encoding="utf-8",
    )
    binary = tmp_path / "source_visibility_ranges"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "23 owner-range cases, 20 dirty replay cases, 4 callback flag cases passed"


def test_owned_dma_wrappers_stage_census_before_raw_delegation():
    wrappers = (
        (
            "rpu_launch_ddr_broadcast_spm_dma",
            "const at::Tensor& ddr_tensor",
            "stage_census_fixed_burst(",
            "GraphOuterFastDmaSide::Source",
        ),
        (
            "rpu_launch_ddr_broadcast_spm_dma_mutable",
            "const at::Tensor& owner_tensor",
            "stage_census_mutable_burst(",
            "GraphOuterFastDmaSide::Source",
        ),
        (
            "rpu_launch_ddr_scatter_spm_dma",
            "const at::Tensor& ddr_tensor",
            "stage_census_fixed_burst(",
            "GraphOuterFastDmaSide::Source",
        ),
        (
            "rpu_launch_ddr_scatter_spm_dma_mutable",
            "const at::Tensor& owner_tensor",
            "stage_census_mutable_burst(",
            "GraphOuterFastDmaSide::Source",
        ),
        (
            "rpu_launch_spm_copy_ddr_dma",
            "const at::Tensor& ddr_tensor",
            "stage_census_fixed_burst(",
            "GraphOuterFastDmaSide::Destination",
        ),
        (
            "rpu_launch_spm_scatter_ddr_dma",
            "const at::Tensor& ddr_tensor",
            "stage_census_fixed_burst(",
            "GraphOuterFastDmaSide::Destination",
        ),
        (
            "rpu_launch_spm_copy_ddr_dma_mutable",
            "const at::Tensor& owner_tensor",
            "stage_census_mutable_burst(",
            "GraphOuterFastDmaSide::Destination",
        ),
    )
    for name, signature, stage_token, side in wrappers:
        body = _squash(
            _overload_body("src/ops/rpu_memcpy.cpp", name, signature)
        )
        validation = body.index("validate_fp16_dma_owner_slice(")
        graph_gate = body.index("if (graph_dma::active())", validation)
        stage = body.index(stage_token, graph_gate)
        delegation = body.index(f"{name}(", stage)
        assert validation < graph_gate < stage < delegation, name
        assert side in body[stage:delegation], name


def test_owned_fixed_spm_scatter_checks_every_size_t_arithmetic_step():
    body = _squash(
        _overload_body(
            "src/ops/rpu_memcpy.cpp",
            "rpu_launch_spm_scatter_ddr_dma",
            "const at::Tensor& ddr_tensor",
        )
    )
    guard = body.index("dst_offset_elements >= 0")
    byte_begin = body.index(
        "const size_t byte_begin = checked_size_t_multiply(", guard
    )
    per_core = body.index(
        "const size_t per_core_bytes = checked_size_t_multiply(", byte_begin
    )
    stride_span = body.index(
        "const size_t stride_span = checked_size_t_multiply(", per_core
    )
    byte_count = body.index(
        "const size_t byte_count = checked_size_t_add(", stride_span
    )
    owner_validation = body.index(
        "validate_fp16_dma_owner_slice(", byte_count
    )
    graph_gate = body.index("if (graph_dma::active())", owner_validation)
    stage = body.index("stage_census_fixed_burst(", graph_gate)
    delegation = body.index("rpu_launch_spm_scatter_ddr_dma(", stage)
    assert (
        guard < byte_begin < per_core < stride_span < byte_count
        < owner_validation < graph_gate < stage < delegation
    )

    multiply = _squash(
        _function_body(
            "src/ops/rpu_memcpy.cpp", "checked_size_t_multiply"
        )
    )
    add = _squash(
        _function_body("src/ops/rpu_memcpy.cpp", "checked_size_t_add")
    )
    assert "std::numeric_limits<size_t>::max()" in multiply
    assert "rhs <= limit / lhs" in multiply
    assert "std::numeric_limits<size_t>::max() - lhs" in add


def test_fmb_outer_fast_component_and_input_publish_after_validation():
    stage = _squash(
        _function_body(
            "src/core/fused_model_base.cpp",
            "FusedModelBase::stage_spm_outer_fast_component",
        )
    )
    publish = stage.index("guard.stage_outer_fast_component(")
    cache_loop = stage.index("for (size_t i = 0; i < k_caches.size(); ++i)")
    stamp_calls = _positions(stage, r"\bowner_stamps\.emplace_back\s*\(")
    assert len(stamp_calls) >= 2
    assert cache_loop < stamp_calls[0] < stamp_calls[1] < publish
    for token in (
        "pimpl_->pipeline_lease_epoch_ != 0",
        "pimpl_->pipeline_plan_hash_ != 0",
        "pimpl_->pipeline_manifest_snapshot_.valid",
        "pimpl_->model_state_gen_ ==",
        "k_caches.size()",
        "v_caches.size()",
        "pimpl_->num_layers_",
        "at::kPrivateUse1",
        "at::kHalf",
        "is_contiguous()",
    ):
        assert token in stage[:publish]

    bind = _squash(
        _function_body(
            "src/core/fused_model_base.cpp",
            "FusedModelBase::bind_spm_outer_fast_input",
        )
    )
    validation = bind.index("TORCH_CHECK(")
    bind_publish = bind.index("guard.bind_fast_mutable_dma(")
    assert validation < bind_publish
    for token in (
        "hidden_states.dim() == 3",
        "hidden_states.size(0) == 1",
        "hidden_states.size(2) == pimpl_->hidden_size_",
        "at::kHalf",
        "at::kPrivateUse1",
        "hidden_states.is_contiguous()",
        "current_hidden_in_src_base()",
        "GraphOuterFastDmaSide::Source",
        "hidden_states.nbytes()",
    ):
        assert token in bind

    invalidate = _squash(
        _function_body(
            "src/core/fused_model_base.cpp",
            "FusedModelBase::invalidate_model_state",
        )
    )
    assert invalidate.index("pimpl_->model_state_gen_++") < invalidate.index(
        "++*pimpl_->outer_fast_model_generation_"
    ) < invalidate.index("retire_pipeline_manifest(")


def test_fmb_composite_mutable_input_slots_are_fixed_and_symmetric():
    impl = _read("src/core/fused_model_base_impl.h")
    assert "std::array<uint64_t, 3> composite_hidden_in_src_bases_" in impl

    selector = _squash(
        _function_body(
            "src/core/fused_model_base.cpp",
            "FusedModelBase::current_hidden_in_src_base",
        )
    )
    assert "if (!runtime.armed) return pimpl_->hidden_in_src_base_" in selector
    assert "runtime.producer_local_ordinal" in selector
    assert "pimpl_->composite_hidden_in_src_bases_[" in selector

    for function in (
        "FusedModelBase::bind_spm_outer_fast_input",
        "FusedModelBase::emit_layer_input_dma",
        "FusedModelBase::emit_spm_pipeline_member_ingress_dma",
        "FusedModelBase::run_all_layers_impl",
    ):
        body = _squash(
            _function_body("src/core/fused_model_base.cpp", function)
        )
        assert "current_hidden_in_src_base()" in body, function

    run = _squash(
        _function_body(
            "src/core/fused_model_base.cpp",
            "FusedModelBase::run_all_layers_impl",
        )
    )
    preload_role = run.index(
        "consume_spm_pipeline_composite_preload_follower()"
    )
    follower_gate = run.index("!composite_preload_follower", preload_role)
    preload_dispatch = run.index("run_preload_callbacks_(", follower_gate)
    assert preload_role < follower_gate < preload_dispatch

    build = _squash(
        _function_body(
            "src/core/fused_model_base.cpp",
            "SpmCompositeTraceCoordinator::begin_build_occurrence",
        )
    )
    replay = _squash(
        _function_body(
            "src/core/fused_model_base.cpp",
            "SpmCompositeTraceCoordinator::begin_replay_occurrence",
        )
    )
    finish = _squash(
        _function_body(
            "src/core/fused_model_base.cpp",
            "SpmCompositeTraceCoordinator::finish_occurrence",
        )
    )
    cancel = _squash(
        _function_body(
            "src/core/fused_model_base.cpp",
            "SpmCompositeTraceCoordinator::cancel_occurrence",
        )
    )
    assert "arm_spm_pipeline_composite_occurrence_runtime(" in build
    assert "arm_spm_pipeline_composite_occurrence_runtime(" in replay
    assert "finish_spm_pipeline_composite_occurrence_runtime(" in finish
    assert "++pimpl_->next_producer_occurrence" in finish
    assert "++pimpl_->next_replay_producer_occurrence" in finish
    assert "cancel_spm_pipeline_composite_occurrence_runtime()" in cancel


def test_direct_dma_contract_is_required_in_the_same_host_ci_invocation():
    direct_path = ROOT / DIRECT_DMA_SOURCE_CONTRACT
    assert direct_path.is_file()
    direct_contract = direct_path.read_text(encoding="utf-8")
    for whole_repo_gate in (
        "test_raw_queue_and_sdk_dma_surface_is_allowlisted",
        "test_direct_dma_guards_precede_every_side_effect",
        "test_immediate_dma_call_inventory_is_exact",
    ):
        assert whole_repo_gate in direct_contract

    ci = _read(".gitlab-ci.yml")
    assert "bash scripts/ci/run_host_tests.sh" in ci
    runner = _read("scripts/ci/run_host_tests.sh")
    assert "set -- tests python/rpu_backend/tests/acceptance" in runner
    for directory in ("core", "graph", "library", "model", "quant"):
        assert f"python/rpu_backend/tests/{directory}" in runner
    assert "python/rpu_backend/tests/ops/test_siglip_pre_embedded_rows.py" in runner
    invocation = runner.replace("\\\n", " ").splitlines()[-1]
    assert '-m pytest -q -ra ' in invocation
    assert invocation.endswith(' "$@"')


def test_host_contracts_imports_package_from_current_checkout():
    runner = _read("scripts/ci/run_host_tests.sh")
    assert 'cd "$CI_PROJECT_DIR"' in runner
    assert 'PYTHONPATH="$CI_PROJECT_DIR:$CI_PROJECT_DIR/python:$RPU_HOST_SITE_PACKAGES"' in runner
    assert "PYTHONNOUSERSITE=1" in runner


def test_semantic_input_dtype_and_ring_proofs_are_fail_closed():
    roles = _cpp_block(
        "src/graph/graph_runtime.h", "enum class GraphDdrRegisterRole"
    )
    proofs = _cpp_block(
        "src/graph/graph_runtime.h", "enum class GraphKernelNoDdrProof"
    )
    assert "SemanticInput = 4" in roles
    assert "Int8ModelWeight = 5" in roles
    assert "PerChannelScale = 6" in roles
    assert "Fp8ModelWeight = 7" in roles
    assert "AllGatherSpm = 2" in proofs
    assert "FillSpm = 4" in proofs
    assert "SliceSpm = 5" in proofs
    assert "RingAllReduceSpmResidual = 6" in proofs
    assert "LegacyAllReduceSpmResidual" not in proofs
    assert "AllReduceSpmResidualV2" not in proofs

    dtype_gate = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp",
            "ddr_register_role_accepts_dtype",
        )
    )
    assert "case GraphDdrRegisterRole::SemanticInput" in dtype_gate
    assert "case GraphDdrRegisterRole::Int8ModelWeight" in dtype_gate
    assert "case GraphDdrRegisterRole::PerChannelScale" in dtype_gate
    assert "case GraphDdrRegisterRole::Fp8ModelWeight" in dtype_gate
    assert "dtype == at::kChar" in dtype_gate
    assert "dtype == at::kByte" in dtype_gate
    assert "dtype == at::kShort" in dtype_gate
    assert "dtype == at::kInt" in dtype_gate
    for legacy_role in ("ModelWeight", "KeyCache", "ValueCache"):
        assert f"case GraphDdrRegisterRole::{legacy_role}" in dtype_gate
    assert "return dtype == at::kHalf" in dtype_gate

    policy = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp", "validate_policy"
        )
    )
    all_gather_proof = policy.index("GraphKernelNoDdrProof::AllGatherSpm")
    fill_proof = policy.index("GraphKernelNoDdrProof::FillSpm")
    slice_proof = policy.index("GraphKernelNoDdrProof::SliceSpm")
    ring_proof = policy.index("GraphKernelNoDdrProof::RingAllReduceSpmResidual")
    for read_only_role in (
        "SemanticInput",
        "Int8ModelWeight",
        "PerChannelScale",
        "Fp8ModelWeight",
    ):
        assert (
            f"abi.role != GraphDdrRegisterRole::{read_only_role}"
        ) in policy
    assert "abi.access == GraphDdrRegisterAccess::Read" in policy
    explicit_branch = policy.index(
        "rule.kind == GraphKernelRegisterRuleKind::IntrinsicNoDdr"
    )
    assert explicit_branch < all_gather_proof < fill_proof < slice_proof < ring_proof
    assert "explicit no-DDR rule needs a launcher proof" in _function_body_raw(
        "src/graph/graph_register_census.cpp", "validate_policy"
    )


def test_dynamic_explicit_no_ddr_requires_staged_name_and_proof():
    header = _squash(_read("src/graph/graph_runtime.h"))
    assert (
        "void stage_kernel_no_ddr(const std::string& name, "
        "GraphKernelNoDdrProof proof);"
    ) in header

    stage = _squash(
        _overload_body(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::stage_kernel_no_ddr",
            "const std::string& name",
        )
    )
    rule = stage.index("const auto& rule = kernel_register_rule_for(name)")
    explicit_kind = stage.index(
        "rule.kind == GraphKernelRegisterRuleKind::ExplicitNoDdr", rule
    )
    proof_match = stage.index("rule.no_ddr_proof == proof", explicit_kind)
    pending_name = stage.index("pending.kernel_name = name", proof_match)
    replay_id = stage.index("!node.kernel_id.has_value()", pending_name)
    replay_name = stage.index("node.kernel_name == name", replay_id)
    assert rule < explicit_kind < proof_match < pending_name < replay_id < replay_name

    admit = _squash(
        _overload_body_raw(
            "src/graph/graph_register_census.cpp",
            "RpuKernelGraph::admit_kernel_register_lookup",
            "const std::string& name",
        )
    )
    intrinsic_only = admit.index(
        "if (!pending_kernel_register_census_.has_value())"
    )
    binary_identity = admit.index(
        'name == "binary_1xC_NxC_tileC"', intrinsic_only
    )
    intrinsic_rule = admit.index(
        "rule.kind == GraphKernelRegisterRuleKind::IntrinsicNoDdr",
        binary_identity,
    )
    staged_or_intrinsic = admit.index(
        "auto& pending = *pending_kernel_register_census_", intrinsic_rule
    )
    exact_pending_name = admit.index("*pending.kernel_name == name")
    lookup_commit = admit.index("pending.kernel_lookup_seen = true")
    assert (
        intrinsic_only
        < binary_identity
        < intrinsic_rule
        < staged_or_intrinsic
        < exact_pending_name
        < lookup_commit
    )
    assert admit.count('name == "binary_1xC_NxC_tileC"') == 1


def test_fill_and_slice_spm_stage_only_after_fail_closed_range_proofs():
    fill = _squash(
        _function_body_raw("src/ops/rpu_memcpy.cpp", "rpu_launch_fill_spm_kernel")
    )
    fill_range = fill.index("candidate - spm_base <=")
    fill_mode = fill.index("constexpr uint16_t mem_flag = 0")
    fill_stage = fill.index(
        'stage_kernel_no_ddr("fill", GraphKernelNoDdrProof::FillSpm)'
    )
    fill_get = fill.index('get_kernel_reset("fill")')
    fill_reg = fill.index("set_regs(5, mem_flag)")
    assert fill_range < fill_mode < fill_stage < fill_get < fill_reg
    assert "SPM_ALLOC.is_initialized()" in fill[:fill_stage]
    assert "SpmAllocator::SPM_USABLE - byte_count" in fill[:fill_stage]

    slice_spm = _squash(
        _function_body_raw(
            "src/ops/rpu_slice.cpp", "rpu_launch_slice_spm_kernel"
        )
    )
    slice_stage = slice_spm.index(
        "stage_kernel_no_ddr(kernel_name, GraphKernelNoDdrProof::SliceSpm)"
    )
    slice_get = slice_spm.index("get_kernel_reset(kernel_name)")
    range_checks = [
        match.start()
        for match in re.finditer(
            r"validate_slice_spm_interval\s*\(", slice_spm
        )
    ]
    assert len(range_checks) == 3
    assert max(range_checks) < slice_stage < slice_get
    for token in (
        "slice_checked_multiply(",
        "slice_checked_add(",
        "last_read_element < input_numel",
        "constexpr uint16_t in256 = 0",
        "constexpr uint16_t out256 = 0",
    ):
        assert token in slice_spm[:slice_stage]
    assert "set_regs(4, in256)" in slice_spm[slice_get:]
    assert "set_regs(5, out256)" in slice_spm[slice_get:]

    slice_source = (ROOT / "src/ops/rpu_slice.cpp").read_text(encoding="utf-8")
    validator = _squash(
        _function_body_raw("src/ops/rpu_slice.cpp", "validate_slice_spm_interval")
    )
    assert "core < SpmAllocator::NUM_CORES" in validator
    assert "SPM_ALLOC.addr(core, 0)" in validator
    assert "capacity - (address - spm_base)" in validator
    assert "core-0 SPM" not in slice_source

    ordinary_slice = _function_body_raw(
        "src/ops/rpu_slice.cpp", "rpu_launch_slice_kernel"
    )
    assert "stage_kernel_no_ddr" not in ordinary_slice
    assert "KernelCache::instance().get_kernel(kernel_name)" in ordinary_slice


def test_semantic_policy_hash_extension_preserves_zero_count_legacy_digest():
    digest = _squash(
        _function_body(
            "src/graph/graph_register_census.cpp", "stable_policy_digest"
        )
    )
    extension = digest.index(
        "if (policy.expected_semantic_input_operand_count != 0)"
    )
    phases = digest.index("for (const auto& phase : policy.phases)")
    assert phases < extension
    assert "UINT64_C(0x53454d504f4c4943)" in digest[extension:]




def test_fmb_post_fn_cursor_is_not_published_inside_typed_census():
    run_all_layers = _squash(
        _function_body_raw(
            "src/core/fused_model_base.cpp",
            "FusedModelBase::run_all_layers_impl",
        )
    )
    census_gate = run_all_layers.index(
        "!RpuKernelGraph::active().kernel_register_census_active()"
    )
    mark = run_all_layers.index("RpuKernelGraph::active().mark_post_fn_cursor()")
    post_fn = run_all_layers.index("std::invoke(static_cfg.post_fn, *this)", mark)
    assert census_gate < mark < post_fn


def test_plain_ddr_writer_resolves_live_addresses_without_snapshot(tmp_path):
    """Run the actual shared validators and plain staging branch without device.

    Tensor/storage stand-ins inject invalid provenance and count metadata
    copies; all validation and address-pair logic is extracted from production.
    """
    import shutil
    import subprocess

    compiler = shutil.which("g++")
    assert compiler is not None, "host contract requires a C++ compiler"
    relative = "src/graph/graph_register_census.cpp"
    functions = (
        ("bool ddr_register_role_accepts_dtype(GraphDdrRegisterRole role, at::ScalarType dtype)",
         "ddr_register_role_accepts_dtype"),
        ("void validate_ddr_register_role_dtype(GraphDdrRegisterRole role, at::ScalarType dtype)",
         "validate_ddr_register_role_dtype"),
        ("void validate_ddr_register_operand(const GraphDdrRegisterOperandSpec& spec)",
         "validate_ddr_register_operand"),
        ("void validate_ddr_register_device_address(uint64_t dev_addr)",
         "validate_ddr_register_device_address"),
        ("void validate_grouped_uint4_metadata_address(const GraphDdrRegisterOperandSpec& spec, uint64_t dev_addr)",
         "validate_grouped_uint4_metadata_address"),
        ("uint64_t resolve_ddr_register_address(const GraphDdrRegisterOperandSpec& spec)",
         "resolve_ddr_register_address"),
        ("GraphDdrRegisterOccurrence snapshot_operand(const GraphDdrRegisterOperandSpec& spec)",
         "snapshot_operand"),
        ("void check_same_occurrence(const GraphDdrRegisterOccurrence& built, const GraphDdrRegisterOccurrence& live, const char* context)",
         "check_same_occurrence"),
    )
    definitions = "\n".join(
        signature + " {\n" + _function_body_raw(relative, name) + "\n}"
        for signature, name in functions
    )
    stage = _function_body_raw(relative, "RpuKernelGraph::stage_kernel_ddr_registers")
    plain, _ = _block_after(stage, "if (!kernel_register_census_active_)")
    # The active branch remains the full owner/provenance path, with no bypass.
    assert "snapshot_operand(operand)" in stage[stage.index("!kernel_register_census_poisoned_"):]
    harness = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
template<class... T> [[noreturn]] void fail(T&&... values) {
    std::ostringstream stream; (stream << ... << values);
    throw std::runtime_error(stream.str());
}
#define TORCH_CHECK(value, ...) do { if (!(value)) fail(__VA_ARGS__); } while (0)
static int owner_copies = 0, storage_copies = 0, metadata_reads = 0, lookups = 0;
static uint64_t live_address = 0x82000000;
namespace c10 {
enum class DeviceType { CPU, PrivateUse1 };
enum class Layout { Strided, Sparse };
struct Device { DeviceType value = DeviceType::PrivateUse1;
    DeviceType type() const { return value; }
    bool operator==(const Device& rhs) const { return value == rhs.value; }
};
std::ostream& operator<<(std::ostream& s, Device d) { return s << int(d.value); }
struct Storage {
    bool present = true; const void* impl = reinterpret_cast<void*>(0x1000);
    const void* pointer = reinterpret_cast<void*>(0x2000);
    size_t bytes = 8192;
    Storage() = default;
    Storage(const Storage& rhs) { *this = rhs; }
    Storage& operator=(const Storage& rhs) {
        ++storage_copies; present=rhs.present; impl=rhs.impl;
        pointer=rhs.pointer; bytes=rhs.bytes; return *this;
    }
    explicit operator bool() const { return present; }
    const void* unsafeGetStorageImpl() const { return impl; }
    const void* data() const { return pointer; }
    size_t nbytes() const { ++metadata_reads; return bytes; }
};
}
namespace at {
enum ScalarType { kHalf, kShort, kInt, kChar, kByte, kFloat };
struct Tensor {
    bool valid = true; c10::Device dev; ScalarType dtype = kChar;
    c10::Layout format = c10::Layout::Strided; c10::Storage backing;
    const void* impl = reinterpret_cast<void*>(0x3000);
    const void* pointer = reinterpret_cast<void*>(0x2000);
    std::vector<int64_t> shape{16,32}, stride{32,1}; int64_t offset=0;
    Tensor() = default;
    Tensor(const Tensor& rhs) { *this = rhs; }
    Tensor& operator=(const Tensor& rhs) {
        ++owner_copies; valid=rhs.valid; dev=rhs.dev; dtype=rhs.dtype;
        format=rhs.format; backing=rhs.backing; impl=rhs.impl;
        pointer=rhs.pointer; shape=rhs.shape; stride=rhs.stride; offset=rhs.offset;
        return *this;
    }
    bool defined() const { return valid; }
    c10::Device device() const { return dev; }
    ScalarType scalar_type() const { return dtype; }
    c10::Layout layout() const { return format; }
    const c10::Storage& storage() const { return backing; }
    const void* unsafeGetTensorImpl() const { return impl; }
    const void* const_data_ptr() const { return pointer; }
    bool is_contiguous() const {
        int64_t step=1;
        for(size_t i=shape.size();i>0;--i) {
            if(shape[i-1]!=1 && stride[i-1]!=step)return false;
            step*=shape[i-1];
        }
        return true;
    }
    const std::vector<int64_t>& sizes() const { ++metadata_reads; return shape; }
    const std::vector<int64_t>& strides() const { ++metadata_reads; return stride; }
    int64_t storage_offset() const { ++metadata_reads; return offset; }
    size_t nbytes() const { ++metadata_reads; return 512; }
};
}
namespace rhino_lkn {
uint64_t RpuGetDevAddr(void*) { ++lookups; return live_address; }
}
enum class GraphDdrRegisterRole { GRAPH_DDR_ROLE_MEMBERS };
enum class GraphDdrRegisterAccess { Read, Write };
enum class GraphDdrRegisterEncoding { DevAddrShift8LoHi, Invalid };
struct GraphDdrRegisterAbi {
    GraphDdrRegisterRole role = GraphDdrRegisterRole::Int8ModelWeight;
    GraphDdrRegisterAccess access = GraphDdrRegisterAccess::Read;
    GraphDdrRegisterEncoding encoding = GraphDdrRegisterEncoding::DevAddrShift8LoHi;
    unsigned reg_lo=6, reg_hi=7;
    bool operator==(const GraphDdrRegisterAbi& rhs) const {
        return role==rhs.role && access==rhs.access && encoding==rhs.encoding &&
            reg_lo==rhs.reg_lo && reg_hi==rhs.reg_hi;
    }
};
struct GraphDdrRegisterOperandSpec { GraphDdrRegisterAbi abi; at::Tensor tensor; };
struct GraphDdrRegisterOccurrence {
    GraphDdrRegisterAbi abi; at::Tensor tensor_owner; c10::Storage storage_owner;
    const void *tensor_impl=nullptr, *storage_impl=nullptr, *data_ptr=nullptr,
               *storage_data_ptr=nullptr;
    at::ScalarType dtype; c10::Device device; c10::Layout layout;
    std::vector<int64_t> sizes, strides; int64_t storage_offset=0;
    size_t tensor_nbytes=0, storage_nbytes=0;
    uint64_t tensor_topology_id=0, dev_addr=0;
};
uint64_t tensor_topology_id(const at::Tensor&, size_t, size_t) {
    ++metadata_reads; return 7;
}
struct GraphKernelRegisterWriter {
    struct RegisterPair { unsigned reg_lo, reg_hi; uint32_t encoded_addr; };
    std::vector<RegisterPair> pairs;
    GraphKernelRegisterWriter(void*, int, std::vector<RegisterPair> values)
        : pairs(std::move(values)) {}
};
static bool has_kernel_register_census_nodes_ = false;
void reset_counts() { owner_copies=storage_copies=metadata_reads=lookups=0; }
std::string caught(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception& e) { return e.what(); }
    return {};
}
'''
    # The extracted production dtype validator handles all published roles.
    # Import its actual enum instead of maintaining a stale subset in the stub.
    harness = harness.replace(
        "GRAPH_DDR_ROLE_MEMBERS",
        _cpp_block("src/graph/graph_runtime.h", "enum class GraphDdrRegisterRole"),
    )
    main = r'''
int main() {
    GraphDdrRegisterOperandSpec good;
    std::vector<GraphDdrRegisterOperandSpec> operands{good};
    for (uint64_t address : {0x82000000ULL,0x84000100ULL,0x82000000ULL}) {
        live_address=address; reset_counts();
        auto writer=stage_plain(operands);
        assert(writer.pairs.size()==1 && writer.pairs[0].reg_lo==6 &&
               writer.pairs[0].reg_hi==7 && writer.pairs[0].encoded_addr==(address>>8));
        assert(lookups==1 && owner_copies==0 && storage_copies==0 && metadata_reads==0);
    }
    has_kernel_register_census_nodes_=true; reset_counts();
    assert(caught([&]{ stage_plain(operands); }).find("required census guard")!=std::string::npos);
    assert(lookups==0 && metadata_reads==0);
    has_kernel_register_census_nodes_=false;
    for (auto role : {GraphDdrRegisterRole::PackedFp4ModelWeight,
                      GraphDdrRegisterRole::Fp8BlockScale}) {
        auto spec=good; spec.abi.role=role; spec.tensor.dtype=at::kByte;
        reset_counts();
        assert(resolve_ddr_register_address(spec)==live_address);
        assert(lookups==1 && metadata_reads==0);
        spec.tensor.dtype=at::kHalf; reset_counts();
        assert(caught([&]{resolve_ddr_register_address(spec);}).find("role/dtype")!=std::string::npos);
        assert(lookups==0);
    }
    for (auto role : {GraphDdrRegisterRole::PackedUint4ModelWeight,
                      GraphDdrRegisterRole::PerGroupScale,
                      GraphDdrRegisterRole::NegativeZeroPoint}) {
        auto spec=good; spec.abi.role=role;
        const bool packed=role==GraphDdrRegisterRole::PackedUint4ModelWeight;
        spec.tensor.dtype=packed ? at::kByte : at::kHalf;
        if(!packed) {
            spec.tensor.shape={256,4,1,8,4,64};
            spec.tensor.stride={8192,2048,2048,256,64,1};
            auto legacy=spec;legacy.tensor.shape={256,16,512};legacy.tensor.stride={8192,512,1};
            assert(caught([&]{resolve_ddr_register_address(legacy);}).find("Qblock4")!=std::string::npos);
            assert(caught([&]{snapshot_operand(legacy);}).find("Qblock4")!=std::string::npos);
            live_address=0x82000100;
            assert(caught([&]{resolve_ddr_register_address(spec);}).find("core_count*512")!=std::string::npos);
            assert(caught([&]{snapshot_operand(spec);}).find("core_count*512")!=std::string::npos);
            live_address=0x82000000;
            auto noncontiguous=spec;noncontiguous.tensor.stride[4]=65;
            assert(caught([&]{resolve_ddr_register_address(noncontiguous);}).find("Qblock4")!=std::string::npos);
        }
        assert(resolve_ddr_register_address(spec)==live_address);
        auto saved=snapshot_operand(spec);
        auto live=snapshot_operand(spec);
        check_same_occurrence(saved,live,"grouped UINT4");
        spec.tensor.dtype=packed ? at::kHalf : at::kByte;
        reset_counts();
        assert(caught([&]{resolve_ddr_register_address(spec);}).find("role/dtype")!=std::string::npos);
        assert(lookups==0);
    }
    using Mutation = std::function<void(GraphDdrRegisterOperandSpec&)>;
    const std::vector<std::pair<Mutation,std::string>> faults = {
        {[](auto& s){s.tensor.valid=false;}, "undefined"},
        {[](auto& s){s.tensor.dev.value=c10::DeviceType::CPU;}, "device-resident"},
        {[](auto& s){s.tensor.dtype=at::kFloat;}, "role/dtype"},
        {[](auto& s){s.tensor.format=c10::Layout::Sparse;}, "strided layout"},
        {[](auto& s){s.abi.encoding=GraphDdrRegisterEncoding::Invalid;}, "unsupported encoding"},
        {[](auto& s){s.abi.reg_lo=7;s.abi.reg_hi=8;}, "even low register"},
        {[](auto& s){s.abi.reg_hi=9;}, "adjacent high register"},
        {[](auto& s){s.tensor.backing.present=false;}, "no StorageImpl"},
        {[](auto& s){s.tensor.impl=nullptr;}, "incomplete storage provenance"},
        {[](auto& s){s.tensor.backing.impl=nullptr;}, "incomplete storage provenance"},
        {[](auto& s){s.tensor.pointer=nullptr;}, "incomplete storage provenance"},
        {[](auto& s){s.tensor.backing.pointer=nullptr;}, "incomplete storage provenance"},
    };
    for (const auto& fault : faults) {
        auto spec=good; fault.first(spec); reset_counts();
        auto plain=caught([&]{resolve_ddr_register_address(spec);});
        assert(plain.find(fault.second)!=std::string::npos && lookups==0);
        auto census=caught([&]{snapshot_operand(spec);});
        assert(census==plain && lookups==0);
    }
    for (uint64_t address : {0ULL,0x82000001ULL,0x10000000000ULL}) {
        live_address=address; reset_counts();
        auto plain=caught([&]{resolve_ddr_register_address(good);});
        assert(!plain.empty() && lookups==1);
        auto census=caught([&]{snapshot_operand(good);});
        assert(census==plain && lookups==2);
    }
    live_address=0x82000000; reset_counts();
    auto built=snapshot_operand(good);
    assert(owner_copies>0 && storage_copies>0 && metadata_reads>0 && lookups==1);
    auto live=snapshot_operand(good);
    check_same_occurrence(built,live,"test");
    good.tensor.stride={1,16};
    live=snapshot_operand(good);
    assert(caught([&]{check_same_occurrence(built,live,"test");}).find("provenance drifted")!=std::string::npos);
    live_address=0x84000000;
    live=snapshot_operand(operands[0]);
    assert(caught([&]{check_same_occurrence(built,live,"test");}).find("provenance drifted")!=std::string::npos);
}
'''
    source = tmp_path / "plain_ddr_writer.cpp"
    source.write_text(
        harness + definitions
        + "\nGraphKernelRegisterWriter stage_plain(const std::vector<GraphDdrRegisterOperandSpec>& operands) {\n"
        + plain + "\n}\n" + main,
        encoding="utf-8",
    )
    binary = tmp_path / "plain_ddr_writer"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
