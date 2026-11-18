"""Host test helpers for the real generated Pi/Wall candidate catalog."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
CATALOG_INCLUDE=f'#include "{ROOT / "src/ops/rpu_pi_wall_tiling.h"}"\n'


def absolute_candidate_includes(source):
    def include(match):
        owners = [ROOT / folder / match[1] for folder in ("src/ops", "src/core")]
        owners = [path for path in owners if path.is_file()]
        assert len(owners) == 1, f"ambiguous or missing generated catalog: {match[1]}"
        return f'#include "{owners[0]}"'
    return re.sub(r'#include "(rpu_[^"]+\.inc)"', include, source)


def assert_current_catalog(*operations):
    header=(ROOT/"src/ops/rpu_pi_wall_tiling.h").read_text()
    catalog=(ROOT/"src/ops/rpu_pi_wall_candidates.inc").read_text()
    cache=(ROOT/"src/core/rpu_kernel_cache.h").read_text()
    preload=(ROOT/"src/core/rpu_kernel_cache.inc").read_text()
    assert '#include "rpu_pi_wall_candidates.inc"' in cache
    assert "rpu_pi_wall::candidate_names" in preload
    for operation in operations:
        assert f"Operation::{operation}," in header
    names=re.findall(r'RPU_PI_WALL_CANDIDATE\([^,]+, "([^"]+)", (\d+)\)',catalog)
    assert len(names)==len(set(name for name,_ in names))
    assert [int(i) for _,i in names]==list(range(len(names)))
    for name,_ in names:
        assert f'"{name}"' in header


def with_actual_linear_contracts(source):
    """Bind extracted CPU fixtures to the current generated host contracts.

    Only the SDK cache owner is omitted: IDs, selectors and tile/grid mappings
    come from the production headers. Unrelated fixtures retain their small enum.
    """
    families = ("g05", "specialized", "geglu", "pi_kv", "qwen3vl")
    needed = [name for name in families if f"rpu_{name}_linear::" in source
              and not re.search(r"namespace\s+rpu_" + name + r"_linear\s*\{", source)]
    uses_catalog = any(token in source for token in (
        "rpu_pl_tiling::", "rpu_pi_wall::", "RPU_PL_CANDIDATE",
        "parallel_linear_kernel_id(", "pi_wall_kernel_id("))
    if not needed and not uses_catalog:
        return source
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    enum = re.search(r"enum class KernelId[^{}]*\{[\s\S]*?\};", cache)[0]
    match = re.search(r"enum class KernelId[^{}]*\{[\s\S]*?\};", source)
    if match:
        source = source[:match.start()] + enum + source[match.end():]
        insertion = match.start() + len(enum)
    else:
        # ATen-backed planner fixtures do not always name a KernelId directly.
        marker = re.search(r"^[ \t]*#define TORCH_CHECK[^\n]*(?:\\\n[^\n]*)*", source, re.M)
        includes = list(re.finditer(r"^[ \t]*#include [^\n]+", source, re.M))
        insertion = marker.end() if marker else includes[-1].end()
        source = source[:insertion] + "\n" + enum + source[insertion:]
        insertion += len(enum) + 1
    macro = re.search(r"^[ \t]*#define TORCH_CHECK[^\n]*(?:\\\n[^\n]*)*", source, re.M)
    if macro:
        insertion = max(insertion, macro.end())
    definitions = []
    for name in needed:
        filename = f"rpu_{name}_linear_contracts.h" if name != "pi_kv" else "rpu_pi_kv_linear_contract.h"
        text = (ROOT / "src/ops" / filename).read_text()
        definitions.append(text.replace('#include "rpu_kernel_cache.h"', '').replace('#pragma once', ''))
    for name in ("parallel_linear_kernel_id", "grouped_parallel_linear_kernel_id", "pi_wall_kernel_id"):
        if name + "(" in source + "\n".join(definitions) and not re.search(r"KernelId\s+" + name + r"\(", source):
            start = cache.index("inline KernelId " + name + "(")
            end = cache.index("\n}", start) + 2
            definitions.insert(0, cache[start:end])
    source = source[:insertion] + "\n" + "\n".join(definitions) + "\n" + source[insertion:]
    source = re.sub(r'#include "(rpu_(?:linear|pi_wall)_tiling.h)"',
                    lambda m: f'#include "{ROOT / "src/ops" / m[1]}"', source)
    if needed and not macro and not re.search(r'#include [<"](?:ATen|torch|c10)/', source):
        source = '#include <stdexcept>\n#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("rejected"); } while (0)\n' + source
    return '#include <cstddef>\n#include <cstdint>\n#include <vector>\n' + CATALOG_INCLUDE + absolute_candidate_includes(source)
