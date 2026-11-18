"""Regression: the Qwen3-VL base dir must be threaded explicitly to the builder
with precedence  explicit arg > runtime_env > bundled default, and a
customer-supplied path must NEVER fall back to the /var/cache/test-models dev mount.

Runs on an device host (importing rpu_backend loads the native .so). The build is
stubbed so no device/checkpoint is needed for the threading assertions.
"""
import os
import tempfile

import rpu_backend  # noqa: F401  (loads the backend)
import rpu_backend.adapters.lingbot_vla_v2 as _pkg
from rpu_backend.api import Lingbot2Policy

DEVMOUNT = "/var/cache/test-models/Qwen3-VL-4B-Instruct"


def _mk_base(tmp, name):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    # minimal valid config.json so isdir + config.json checks pass
    with open(os.path.join(d, "config.json"), "w") as f:
        f.write('{"model_type": "qwen3_vl"}')
    return d


def _thread_value(runtime_env=None, qwen3vl_base=None):
    """Return the qwen3vl_base value that .to('rpu') hands to build_lingbot_vla_v2."""
    captured = {}

    class _Stop(Exception):
        pass

    def _fake_build(ckpt, qwen3vl_base=None, **kw):
        captured["v"] = qwen3vl_base
        raise _Stop()

    # Shadow the lazy __getattr__ only for this call. Leaving the fake bound on
    # the package makes later tests exercise the stub instead of the real builder.
    had_override = "build_lingbot_vla_v2" in _pkg.__dict__
    old_override = _pkg.__dict__.get("build_lingbot_vla_v2")
    _pkg.build_lingbot_vla_v2 = _fake_build
    try:
        env = dict(runtime_env or {})
        kw = {"dtype": "fp16", "runtime_env": env}
        if qwen3vl_base:
            kw["qwen3vl_base"] = qwen3vl_base
        pol = Lingbot2Policy.from_checkpoint("/tmp/nonexistent_ckpt", **kw)
        try:
            pol.to("rpu")
        except _Stop:
            pass
    finally:
        if had_override:
            _pkg.build_lingbot_vla_v2 = old_override
        else:
            del _pkg.build_lingbot_vla_v2
    return captured.get("v", "<not-called>")


def test_runtime_env_is_threaded_and_never_hits_devmount():
    with tempfile.TemporaryDirectory() as tmp:
        custom = _mk_base(tmp, "custom")
        v = _thread_value(runtime_env={"RPU_LINGBOT2_QWEN3VL_BASE": custom})
        assert v == os.path.abspath(custom), v
        assert DEVMOUNT not in str(v)


def test_default_used_only_when_nothing_passed():
    v = _thread_value(runtime_env={})
    assert v is None, v  # None ⇒ builder uses the wheel-bundled default


def test_explicit_arg_beats_runtime_env():
    with tempfile.TemporaryDirectory() as tmp:
        custom = _mk_base(tmp, "custom")
        explicit = _mk_base(tmp, "explicit")
        v = _thread_value(runtime_env={"RPU_LINGBOT2_QWEN3VL_BASE": custom},
                          qwen3vl_base=explicit)
        assert v == os.path.abspath(explicit), v


def test_builder_error_reports_source_and_value():
    from rpu_backend.adapters.lingbot_vla_v2 import build_lingbot_vla_v2
    bad = "/tmp/does_not_exist_qwenbase"
    try:
        build_lingbot_vla_v2("/tmp/nonexistent_ckpt", qwen3vl_base=bad)
    except FileNotFoundError as e:
        msg = str(e)
        assert "explicit qwen3vl_base arg" in msg and bad in msg, msg
    else:
        raise AssertionError("expected FileNotFoundError for a bad qwen3vl_base dir")


if __name__ == "__main__":
    test_runtime_env_is_threaded_and_never_hits_devmount()
    test_default_used_only_when_nothing_passed()
    test_explicit_arg_beats_runtime_env()
    test_builder_error_reports_source_and_value()
    print("REGRESSION_OK")
