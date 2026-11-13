"""Unit tests for DCO sign-off, public hygiene, and external plugin SDK gates."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tomllib
import types
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_dco = _load_module("check_dco_under_test", ROOT / "scripts/ci/check_dco.py")
check_hygiene = _load_module("check_hygiene_under_test", ROOT / "scripts/check_public_hygiene.py")


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)


def _commit_file(repo: Path, filename: str, content: str, message: str) -> str:
    target = repo / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", filename], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", message], check=True)
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _load_registry_module(monkeypatch):
    """Load rpu_backend.runtime.registry with or without torch installed."""
    try:
        from rpu_backend.runtime import registry
        return registry
    except ModuleNotFoundError:
        pkg = types.ModuleType("rpu_backend")
        pkg.__path__ = [str(ROOT / "python/rpu_backend")]
        runtime_pkg = types.ModuleType("rpu_backend.runtime")
        runtime_pkg.__path__ = [str(ROOT / "python/rpu_backend/runtime")]

        class UnsupportedModelError(RuntimeError):
            pass

        runtime_pkg.UnsupportedModelError = UnsupportedModelError
        monkeypatch.setitem(sys.modules, "rpu_backend", pkg)
        monkeypatch.setitem(sys.modules, "rpu_backend.runtime", runtime_pkg)
        spec = importlib.util.spec_from_file_location(
            "rpu_backend.runtime.registry",
            ROOT / "python/rpu_backend/runtime/registry.py",
        )
        assert spec is not None and spec.loader is not None
        registry = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "rpu_backend.runtime.registry", registry)
        spec.loader.exec_module(registry)
        return registry


def test_dco_self_test_cli():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci/check_dco.py"), "--self-test"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_dco_range_accepts_public_signoff_and_skips_merges(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    base = _commit_file(
        repo,
        "README.md",
        "init\n",
        "chore: initial commit\n\nSigned-off-by: Alice Public <alice@example.com>",
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "feature"], check=True)
    _commit_file(
        repo,
        "feature.txt",
        "v1\n",
        "feat: add feature\n\nSigned-off-by: xup16 <xuping908@gmail.com>",
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "main"], check=True)
    _commit_file(
        repo,
        "main.txt",
        "m1\n",
        "docs: update main\n\nSigned-off-by: Alice Public <alice@example.com>",
    )
    # Merge commit intentionally carries no Signed-off-by trailer.
    subprocess.run(
        ["git", "-C", str(repo), "merge", "--no-ff", "-q", "feature", "-m", "Merge branch 'feature'"],
        check=True,
    )
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    checked, failures = check_dco.check_commits(repo, f"{base}..{head}")
    assert checked == 2
    assert failures == []
    # HEAD itself is a merge commit, so single-HEAD mode skips it cleanly.
    head_checked, head_failures = check_dco.check_commits(repo)
    assert head_checked == 0 and head_failures == []


def test_dco_rejects_missing_signoff_and_internal_domain_in_git_range(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    base = _commit_file(
        repo,
        "a.txt",
        "1\n",
        "chore: base\n\nSigned-off-by: Alice Public <alice@example.com>",
    )
    missing_sha = _commit_file(repo, "b.txt", "2\n", "feat: forgot signoff")
    corp_email = "engineer@" + ".".join(("rhino", "auto"))
    corp_sha = _commit_file(
        repo,
        "c.txt",
        "3\n",
        f"feat: wrong email\n\nSigned-off-by: Engineer <{corp_email}>",
    )
    cli = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci/check_dco.py"), "--repo", str(repo), "--range", f"{base}..{corp_sha}"],
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 1
    assert missing_sha[:12] in cli.stderr
    assert corp_sha[:12] in cli.stderr
    assert "internal domain" in cli.stderr


def test_public_hygiene_self_test_and_live_repository_pass():
    selftest = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_public_hygiene.py"), "--self-test"],
        capture_output=True,
        text=True,
    )
    assert selftest.returncode == 0, selftest.stderr
    live = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_public_hygiene.py"), "--root", str(ROOT)],
        capture_output=True,
        text=True,
    )
    assert live.returncode == 0, live.stderr


def test_public_hygiene_flags_each_forbidden_category_in_temp_repo(tmp_path):
    repo = tmp_path / "hygiene-repo"
    _init_git_repo(repo)
    bad_lines = "\n".join([
        "ip = '" + ".".join(("10", "10", "88", "24")) + "'",
        "host1 = '" + "-".join(("hx", "desktop")) + "'",
        "host2 = '" + "-".join(("rpu", "nas", "01")) + "'",
        "mail = 'bot@" + ".".join(("rhino", "auto")) + "'",
        "mount = '/" + "-".join(("nfs3", "sh")) + "/data'",
        "plan = '." + "planning/STATE.md'",
        "note = '" + " ".join(("codex", "thread")) + " 0195aaaa-bbbb-cccc-dddd-eeeeeeeeeeee'",
    ]) + "\n"
    _commit_file(repo, "leaks.txt", bad_lines, "test\n\nSigned-off-by: Test <test@example.com>")
    _commit_file(
        repo,
        "clean.txt",
        "public_ip = '192.0.2.5'\nother_net = '10.20.0.1'\nemail = 'dev@example.com'\n",
        "clean\n\nSigned-off-by: Test <test@example.com>",
    )
    scanned, violations = check_hygiene.scan_repo(repo)
    assert scanned == 2
    assert [v[0] for v in violations] == ["leaks.txt"] * 7
    assert {v[2] for v in violations} == {
        "private_lab_ip",
        "internal_hostname",
        "internal_email",
        "internal_nfs_path",
        "planning_directory",
        "agent_thread_id",
    }


def test_plugin_sdk_template_and_discover_plugins_contract(monkeypatch):
    root_pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert root_pyproject["project"]["entry-points"]["rpu_backend.plugins"] == {}

    template_dir = ROOT / "examples/custom_plugin_template"
    template_pyproject = tomllib.loads((template_dir / "pyproject.toml.example").read_text(encoding="utf-8"))
    ep_table = template_pyproject["project"]["entry-points"]["rpu_backend.plugins"]
    assert ep_table == {"my_rhino_plugin": "my_rhino_plugin:register"}

    csrc_cmake = (template_dir / "csrc/CMakeLists.txt.example").read_text(encoding="utf-8")
    csrc_cpp = (template_dir / "csrc/custom_ops_fragment.cpp.example").read_text(encoding="utf-8")
    assert "find_package(Torch REQUIRED)" in csrc_cmake
    assert "TORCH_LIBRARY_FRAGMENT(rpu, m)" in csrc_cpp
    assert "TORCH_LIBRARY_IMPL(rpu, PrivateUse1, m)" in csrc_cpp

    plugin_mod = _load_module(
        "my_rhino_plugin",
        template_dir / "my_rhino_plugin/__init__.py",
    )
    registry = _load_registry_module(monkeypatch)
    saved_adapters = dict(registry.ADAPTERS)
    saved_discovered = registry._PLUGINS_DISCOVERED
    try:
        registry.ADAPTERS.clear()
        registry.register_builtins([("Qwen3ForCausalLM", "rpu_backend.adapters.qwen3", "Qwen3Adapter")])
        registry._PLUGINS_DISCOVERED = False

        ep = SimpleNamespace(
            name="my_rhino_plugin",
            group="rpu_backend.plugins",
            load=lambda: plugin_mod.register,
        )
        monkeypatch.setattr(
            "importlib.metadata.entry_points",
            lambda group=None: [ep] if group == "rpu_backend.plugins" else [],
        )
        registry.discover_plugins()
        resolved = registry.get_adapter(plugin_mod.ARCH_NAME)
        assert resolved is plugin_mod.MyCustomAdapter
        resolved.preflight(SimpleNamespace(hidden_size=1024))
        with pytest.raises(ValueError, match="positive hidden_size"):
            resolved.preflight(SimpleNamespace(hidden_size=0))
        assert resolved.load_native_ops(template_dir / "nonexistent.so") is False

        # Built-in shadowing via entry-point must always fail closed (REG-03).
        shadow_ep = SimpleNamespace(
            name="shadow_plugin",
            group="rpu_backend.plugins",
            load=lambda: (lambda api: api.register_adapter("Qwen3ForCausalLM", plugin_mod.MyCustomAdapter)),
        )
        monkeypatch.setattr(
            "importlib.metadata.entry_points",
            lambda group=None: [shadow_ep] if group == "rpu_backend.plugins" else [],
        )
        with pytest.raises(RuntimeError, match="built-in"):
            registry.discover_plugins()
    finally:
        registry.ADAPTERS.clear()
        registry.ADAPTERS.update(saved_adapters)
        registry._PLUGINS_DISCOVERED = saved_discovered
