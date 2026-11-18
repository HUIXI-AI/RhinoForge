from __future__ import annotations
from typing import Iterable
SCHEMA = 'rpu-fmb-route-evidence.v2'
_LIFECYCLES = {1: 'RETAINED_CACHE', 2: 'BOUNDED_ONESHOT', 3: 'NATIVE_COMPOSITE1', 4: 'COMPOSITE_CHILD'}
_MASK64 = (1 << 64) - 1

def stable_replay(cache, before, after, expected_calls, *, allow_idle=False):
    previous = {entry.graph_lifetime_id: entry for entry in before}
    current = {entry.graph_lifetime_id: entry for entry in after}
    if not previous or len(previous) != len(before) or len(current) != len(after) or (set(previous) != set(current)) or (not cache.cache_invariant_ok()):
        return False
    executed = 0
    for (lifetime, entry) in current.items():
        old = previous[lifetime]
        if lifetime <= 0 or entry.build_generation <= 0 or entry.non_replayable_reason or (entry.kernel_count <= 0):
            return False
        if allow_idle and entry.replay_count == old.replay_count:
            fields = ('build_generation', 'execution_ordinal', 'recapture_count', 'non_replayable_reason')
            if any((getattr(entry, field) != getattr(old, field) for field in fields)):
                return False
            continue
        if entry.build_generation != old.build_generation or entry.recapture_count != old.recapture_count or entry.replay_count <= old.replay_count or (entry.execution_ordinal < old.execution_ordinal):
            return False
        executed += entry.replay_count - old.replay_count
    return executed == expected_calls

def _mix_u64(value_hash: int, value: int) -> int:
    encoded = value & _MASK64
    for shift in range(0, 64, 8):
        value_hash ^= encoded >> shift & 255
        value_hash = value_hash * 1099511628211 & _MASK64
    return value_hash

def _manifest_fingerprint(header: Iterable[int], routes: list[dict[str, object]]) -> int:
    value_hash = _mix_u64(1469598103934665603, 5065778132997921074)
    for value in header:
        value_hash = _mix_u64(value_hash, value)
    for route in routes:
        for value in (route['family'], route['site_id'], route['selector'], route['flags'], len(route['arguments']), *route['arguments']):
            value_hash = _mix_u64(value_hash, int(value))
        invocation = int(route['invocation'])
        if invocation:
            value_hash = _mix_u64(value_hash, invocation)
    return value_hash or 1

def complete_descriptor_evidence(descriptor: Iterable[int], *, expected_lifecycle: str, component_path: str, stage: str) -> dict[str, object]:
    if not all((isinstance(value, str) and value for value in (component_path, stage))):
        raise AssertionError('descriptor component_path and stage must be explicit')
    words = tuple((int(value) for value in descriptor))
    if len(words) < 26 or words[0] not in (3, 4) or words[1] != len(words):
        raise AssertionError('hardware closure requires a complete v3/v4 stage descriptor')
    physical_end = len(words) - (2 if words[0] == 4 else 0)
    counts = words[11:14]
    span_count = words[14]
    if any((value <= 0 for value in (*counts, span_count))):
        raise AssertionError('stage descriptor has invalid chunk/span counts')
    cursor = 15 + 4 * sum(counts) + 3 * span_count
    if cursor + 11 > physical_end:
        raise AssertionError('stage descriptor physical manifest is truncated')
    (state, logical_length, physical_length, execution_padding_rows, kv_logical_length, kv_insert_physical_rows, lifecycle, linear_accumulation, route_count, fingerprint_hi, fingerprint_lo) = words[cursor:cursor + 11]
    cursor += 11
    if state != 1 or route_count <= 0:
        raise AssertionError('hardware closure requires a COMPLETE physical manifest')
    lifecycle_name = _LIFECYCLES.get(lifecycle)
    if lifecycle_name != expected_lifecycle:
        raise AssertionError(f'physical lifecycle {lifecycle_name!r} != {expected_lifecycle!r}')
    routes: list[dict[str, object]] = []
    previous = (-1, -1, -1)
    for _ in range(route_count):
        if cursor + 6 > physical_end:
            raise AssertionError('physical route header is truncated')
        (family, site_id, selector, flags, argument_count, invocation) = words[cursor:cursor + 6]
        cursor += 6
        if argument_count < 0 or cursor + argument_count > physical_end:
            raise AssertionError('physical route arguments are truncated')
        arguments = list(words[cursor:cursor + argument_count])
        cursor += argument_count
        identity = (family, site_id, invocation)
        if family not in range(1, 11) or site_id <= 0 or selector <= 0 or (flags < 0) or (invocation < 0) or any((value < 0 for value in arguments)) or (identity <= previous):
            raise AssertionError('physical route is invalid or non-canonical')
        routes.append({'family': family, 'site_id': site_id, 'invocation': invocation, 'selector': selector, 'flags': flags, 'arguments': arguments})
        previous = identity
    if cursor != physical_end:
        raise AssertionError('stage descriptor has trailing words')
    ranking = {}
    if words[0] == 4:
        policy, mixer_calls = words[-2:]
        if policy not in (1, 2) or not counts[2] <= mixer_calls <= physical_length:
            raise AssertionError('stage descriptor has invalid ranking policy or mixer count')
        ranking = {'ranking_policy': policy, 'ranking_mixer_calls': mixer_calls}
    manifest_header = (state, logical_length, physical_length, execution_padding_rows, kv_logical_length, kv_insert_physical_rows, lifecycle, linear_accumulation, route_count)
    fingerprint = fingerprint_hi << 32 | fingerprint_lo
    if fingerprint != _manifest_fingerprint(manifest_header, routes):
        raise AssertionError('physical manifest fingerprint is stale')
    return {'schema': SCHEMA, 'descriptor_version': words[0], **ranking, 'component_path': component_path, 'stage': stage, 'manifest': {'state': 'COMPLETE', 'logical_length': logical_length, 'physical_length': physical_length, 'execution_padding_rows': execution_padding_rows, 'kv_logical_length': kv_logical_length, 'kv_insert_physical_rows': kv_insert_physical_rows, 'graph_lifecycle': lifecycle_name, 'linear_accumulation': linear_accumulation, 'route_count': route_count, 'fingerprint': f'{fingerprint:016x}'}, 'routes': routes}
