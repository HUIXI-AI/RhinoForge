from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_host_only_import_is_silent() -> None:
    site_packages = [path for path in sys.path if "site-packages" in path]
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'python')!r}); "
        f"sys.path.extend({site_packages!r}); "
        "import rpu_backend, torch; "
        "assert rpu_backend.__version__; "
        "assert hasattr(torch, 'rpu')"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("RPU_BACKEND_SO", None)
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_nofile_preflight_is_silent_and_best_effort(monkeypatch, capsys) -> None:
    from rpu_backend import _bootstrap_env

    calls = []
    monkeypatch.setattr(
        _bootstrap_env.resource,
        "getrlimit",
        lambda _kind: (1024, 4096),
    )
    monkeypatch.setattr(
        _bootstrap_env.resource,
        "setrlimit",
        lambda kind, value: calls.append((kind, value)),
    )
    _bootstrap_env.ensure_nofile_limit()
    assert calls == [
        (_bootstrap_env.resource.RLIMIT_NOFILE, (4096, 4096))
    ]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""

    def deny_limit_change(*_args):
        raise OSError("denied")

    monkeypatch.setattr(
        _bootstrap_env.resource,
        "setrlimit",
        deny_limit_change,
    )
    _bootstrap_env.ensure_nofile_limit()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_public_api_and_host_graph_lifecycle() -> None:
    import rpu_backend
    from rpu_backend.graph import GraphCache, GraphSignature

    assert rpu_backend.__all__ == [
        "__version__",
        "RPUCache",
        "RPUModelForCausalLM",
        "reset_graph_cache",
    ]

    cache = GraphCache()
    signature = GraphSignature(op_id="host-contract", shapes=(1, 16))
    assert cache.phase == "CONFIGURING"
    assert cache.size() == 0
    assert cache.cache_invariant_ok()
    cache.begin_warmup()
    with cache.capture(signature):
        pass
    assert cache.phase == "WARMING"
    with pytest.raises(RuntimeError, match=r"requires the C\+\+ backend"):
        cache.freeze()


def test_external_adapter_registration() -> None:
    from rpu_backend.runtime import registry

    class PublicTestAdapter:
        pass

    architecture = "RhinoForgePublicTestArchitecture"
    try:
        registry.register_adapter(architecture, PublicTestAdapter)
        assert registry.get_adapter(architecture) is PublicTestAdapter
        assert architecture in registry.list_adapters()
        registry.register_adapter(architecture, PublicTestAdapter)
        with pytest.raises(RuntimeError, match="already registered"):
            registry.register_adapter(architecture, type("OtherAdapter", (), {}))
    finally:
        registry.ADAPTERS.pop(architecture, None)


def test_registry_excludes_unsupported_profiles() -> None:
    from rpu_backend import model_registry

    required = {
        "qwen3-0.6b",
        "llama-3.2-1b",
        "qwen3-vl-2b",
        "qwen3-vl-4b",
        "qwen3_5-0.8b",
        "dinov3-vit-b",
        "pi05-libero-finetuned",
        "qwen3-14b-w8a16-lmhead-int8",
    }
    excluded = {
        "qwen3-vl-8b",
        "dinov3-vit-s",
        "dinov3-vit-l",
        "gemma2",
        "paligemma2",
        "paligemma2-3b-pt-224",
        "paligemma2-3b-mix-224",
        "lingbot-vla-4b",
        "qwen3-14b",
        "qwen3-0.6b-w8a16-lmhead-int8",
        "qwen3-1.7b-w8a16-lmhead-int8",
        "qwen3-4b-w8a16-lmhead-int8",
        "qwen3-8b-w8a16-lmhead-int8",
        "qwen3-0.6b-instruct",
        "qwen3-4b-instruct",
    }
    assert required <= model_registry.MODELS.keys()
    assert excluded.isdisjoint(model_registry.MODELS)
    for relative in model_registry.MODELS.values():
        path = Path(relative)
        assert not path.is_absolute()
        assert ".." not in path.parts


def test_qwen3_vl_deepstack_targets_first_text_layers() -> None:
    from types import SimpleNamespace

    from rpu_backend.adapters.qwen3_vl.text import (
        _deepstack_text_layer_indices,
    )

    config = SimpleNamespace(deepstack_visual_indexes=[5, 11, 17])
    assert _deepstack_text_layer_indices(config) == [0, 1, 2]


def test_native_loader_only_selects_packaged_extension(
    tmp_path: Path, monkeypatch
) -> None:
    from rpu_backend._native_loader import find_shared_object

    package_dir = tmp_path / "python" / "rpu_backend"
    package_dir.mkdir(parents=True)
    external = tmp_path / "build" / "rpu_backend.so"
    external.parent.mkdir()
    external.write_bytes(b"external")
    monkeypatch.setenv("RPU_BACKEND_SO", str(external))

    assert find_shared_object(package_dir=str(package_dir)) is None
    packaged = package_dir / "rpu_backend.so"
    packaged.write_bytes(b"packaged")
    assert find_shared_object(package_dir=str(package_dir)) == str(packaged)


def test_rhinovla_has_no_model_source_execution_loader() -> None:
    package = ROOT / "python" / "rpu_backend" / "adapters" / "rhinovla"
    assert not (package / "checkpoint.py").exists()
    assert "load_action_bundle" not in (package / "__init__.py").read_text(
        encoding="utf-8"
    )
