"""Validation shared by policy-scoped ``runtime_env`` mappings."""

from __future__ import annotations

from collections.abc import Collection, Mapping


LKN_CAPACITY_ENV = frozenset({
    "LKN_MAX_BATCH_ENTRIES",
    "LKN_KD_BUF_MB",
    "LKN_INSTR_BUF_MB",
})


def normalize_runtime_env(
    runtime_env: Mapping[str, str] | None,
    *,
    owner: str,
    allowed: Collection[str],
    preimport_only: Collection[str] = (),
) -> dict[str, str]:
    """Return a string mapping after enforcing one policy's exact key set."""
    if runtime_env is None:
        return {}
    if not isinstance(runtime_env, Mapping):
        raise ValueError(f"{owner} runtime_env must be a mapping")

    result: dict[str, str] = {}
    for key, value in runtime_env.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\0" in key
        ):
            raise ValueError(
                f"{owner} runtime_env has invalid environment key {key!r}"
            )
        if key not in allowed:
            if key in preimport_only:
                raise ValueError(
                    f"{owner} runtime_env key {key!r} must be set before "
                    "importing rpu_backend"
                )
            raise ValueError(
                f"{owner} runtime_env has unsupported environment key {key!r}"
            )
        rendered = str(value)
        if "\0" in rendered:
            raise ValueError(
                f"{owner} runtime_env[{key!r}] contains a null byte"
            )
        result[key] = rendered
    return result
