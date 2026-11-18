"""Execute GR00T consumers with the exact fields exported by graph_pybind.cpp."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'python/rpu_backend/adapters/gr00t/runtime.py'
CANARY = ROOT / 'python/rpu_backend/tests/model/gr00t/test_gr00t_qwen3vl_spm_z1_production_canary.py'


def _fields(cpp_type):
    source = (ROOT/'src/graph/graph_pybind.cpp').read_text()
    start = source.index('py::class_<' + cpp_type + '>')
    body = source[start:source.index(';', start)]
    fields = tuple(re.findall(r'\.def_(?:property_)?readonly\("([^"]+)"', body))
    assert fields and len(fields) == len(set(fields))
    return fields


def _bound(cpp_type, **values):
    # __slots__ prevents a fixture from inventing a C++ field that is not bound.
    cls = type(cpp_type, (), {'__slots__': _fields(cpp_type)})
    value = cls()
    for name in cls.__slots__:
        setattr(value, name, values.pop(name, 0))
    assert not values, f'unbound fixture fields: {values}'
    return value


def _identity(source=None):
    tree = ast.parse(RUNTIME.read_text() if source is None else source)
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == '_qwen_z1_graph_identity')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNTIME), 'exec'), namespace)
    return namespace[function.name]


def _case():
    common = dict(graph_lifetime_id=21, build_generation=4, execution_ordinal=3,
                  kernel_count=5, segment_count=1)
    stats = _bound('GraphStats', **common)
    entry = _bound('RpuGraphCache::Snapshot', **common, replay_count=2,
                   recapture_count=0, non_replayable_reason='')
    rows = [entry]
    owner = SimpleNamespace(_qwen_z1_gc=SimpleNamespace(snapshot=lambda: tuple(rows)))
    graph = SimpleNamespace(state=lambda: 2, replayable=lambda: True, debug_stats=lambda: stats)
    return owner, graph, stats, entry, rows


def test_actual_binding_identity_uses_authenticated_snapshot_reason():
    owner, graph, stats, entry, _ = _case()
    assert 'non_replayable_reason' not in _fields('GraphStats')
    assert 'non_replayable_reason' in _fields('RpuGraphCache::Snapshot')
    assert _identity()(owner, graph) == (21, 4, 3, 2, 0, 5, 1)
    mirror = (ROOT/'src/graph/graph_infra.cpp').read_text()
    assert re.search(r's\.non_replayable_reason\s*=\s*gs\.non_replayable_reason;', mirror)
    # The former consumer fails with the actual binding schema, even though
    # its synthetic SimpleNamespace fixture previously supplied this field.
    old = RUNTIME.read_text().replace('or entries[0].non_replayable_reason)',
                                      'or stats.non_replayable_reason)')
    assert old != RUNTIME.read_text()
    with pytest.raises(AttributeError, match='non_replayable_reason'):
        _identity(old)(owner, graph)


@pytest.mark.parametrize('fault', ['reason', 'missing_reason', 'duplicate', 'foreign',
                                  'generation', 'ordinal', 'kernel', 'segment',
                                  'recapture', 'not_replayable'])
def test_identity_rejects_incomplete_or_foreign_bound_evidence(fault):
    owner, graph, stats, entry, rows = _case()
    if fault == 'reason': entry.non_replayable_reason = 'foreign control flow'
    elif fault == 'missing_reason': del entry.non_replayable_reason
    elif fault == 'duplicate': rows.append(entry)
    elif fault == 'foreign': entry.graph_lifetime_id += 1
    elif fault == 'generation': entry.build_generation += 1
    elif fault == 'ordinal': entry.execution_ordinal += 1
    elif fault == 'kernel': entry.kernel_count += 1
    elif fault == 'segment': entry.segment_count += 1
    elif fault == 'recapture': entry.recapture_count = 1
    elif fault == 'not_replayable': graph.replayable = lambda: False
    with pytest.raises((RuntimeError, AttributeError)):
        _identity()(owner, graph)


def test_all_gr00t_debug_stats_consumers_use_actual_bound_fields():
    tree = ast.parse(RUNTIME.read_text())
    consumers = {}
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        aliases = {target.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and n.value.func.attr == 'debug_stats' for target in n.targets if isinstance(target, ast.Name)}
        if not aliases: continue
        fields = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
                  and isinstance(n.value, ast.Name) and n.value.id in aliases}
        assert fields <= set(_fields('GraphStats')), (fn.name, fields - set(_fields('GraphStats')))
        consumers[fn.name] = fields
    assert set(consumers) == {'_qwen_z1_graph_identity', '_qwen_z1_recorded_ranges', '_backbone_qwen_z1'}


