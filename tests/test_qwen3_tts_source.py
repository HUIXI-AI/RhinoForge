"""Board-free checks for reviewed-source admission and CPU-loader ordering."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def package(monkeypatch):
    root = Path(__file__).parents[1] / "python/rpu_backend/adapters/qwen3_tts"
    name = "_test_qwen3_tts_source"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    yield module
    for key in list(sys.modules):
        if key.startswith(name + "."):
            sys.modules.pop(key, None)


@pytest.fixture
def reviewed_source(tmp_path, package, monkeypatch):
    source_module = sys.modules[package.__name__ + ".source"]
    upstream = tmp_path / "upstream"
    path = upstream / "qwen_tts/__init__.py"
    path.parent.mkdir(parents=True)
    # Preparation must never execute the candidate's Python code.
    content = b"raise RuntimeError('must not be imported during preparation')\n"
    path.write_bytes(content)
    hashes = {"__init__.py": hashlib.sha256(content).hexdigest()}
    monkeypatch.setattr(source_module, "SOURCE_HASHES", hashes)
    monkeypatch.setattr(source_module, "PREPARED_HASHES", hashes)
    return upstream, path, content


def test_prepare_preserves_upstream_and_rejects_changes_before_writing(tmp_path, package, reviewed_source):
    upstream, path, original = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    assert path.read_bytes() == original == (prepared / "__init__.py").read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        package.prepare_source(upstream, upstream / "nested")
    path.write_bytes(original + b"# changed\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        package.prepare_source(upstream, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("extra", ["unreviewed.py", "__pycache__/__init__.pyc", "extension.so"])
def test_prepared_package_rejects_unreviewed_import_code(tmp_path, package, reviewed_source, extra):
    upstream, _, _ = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    unexpected = prepared / extra
    unexpected.parent.mkdir(exist_ok=True)
    unexpected.write_bytes(b"unreviewed")
    with pytest.raises(ValueError, match="unreviewed code"):
        package.preflight(tmp_path / "checkpoint", prepared)


@pytest.mark.parametrize("version", ["5.3.0", "5.19.0"])
def test_bad_metadata_fails_before_weight_read_or_source_import(tmp_path, package, reviewed_source, monkeypatch, version):
    upstream, _, _ = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    loader = sys.modules[package.__name__ + ".loader"]
    monkeypatch.setattr(loader.importlib.metadata, "version", lambda _: version)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setattr(loader, "CHECKPOINT_PROFILES", {"0.6b": {"revision": "frozen",
        "hashes": {"config.json": "invalid", "model.safetensors": "never read"}}})
    with pytest.raises(ValueError, match="config.json"):
        package.load_cpu_model(checkpoint, prepared)
    assert not any(key.startswith("_rhinoforge_reviewed_qwen3_tts_") for key in sys.modules)


@pytest.mark.parametrize("version", ["4.57.3", "5.18.0", "5.19.1"])
def test_wrong_transformers_rejected_before_loading(tmp_path, package, reviewed_source, monkeypatch, version):
    upstream, _, _ = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    loader = sys.modules[package.__name__ + ".loader"]
    monkeypatch.setattr(loader.importlib.metadata, "version", lambda _: version)
    with pytest.raises(ValueError, match="transformers==5.3.0"):
        package.load_cpu_model(tmp_path / "absent", prepared)


def test_prepared_source_hash_is_checked_without_import(tmp_path, package, reviewed_source):
    upstream, _, _ = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    (prepared / "__init__.py").write_text("raise RuntimeError('unreviewed')")
    with pytest.raises(ValueError, match="prepared source identity mismatch"):
        package.load_cpu_model(tmp_path / "absent", prepared)


@pytest.mark.parametrize("profile", ["0.6b", "1.7b"])
def test_preflight_selects_whole_checkpoint_and_checks_metadata_before_weights(
        tmp_path, package, reviewed_source, monkeypatch, profile):
    upstream, _, _ = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    loader = sys.modules[package.__name__ + ".loader"]
    monkeypatch.setattr(loader.importlib.metadata, "version", lambda _: "5.3.0")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    profiles = {name: {"revision": name + "-revision", "hashes": {
        "model.safetensors": name + "-weight", "config.json": name + "-config",
        "speech_tokenizer/model.safetensors": "codec-weight", "tokenizer_config.json": "metadata"}}
        for name in ("0.6b", "1.7b")}
    monkeypatch.setattr(loader, "CHECKPOINT_PROFILES", profiles)
    reads = []
    def sha256(path):
        name = str(path.relative_to(checkpoint))
        reads.append(name)
        return profiles[profile]["hashes"][name]
    monkeypatch.setattr(loader, "sha256", sha256)
    result = package.preflight(checkpoint, prepared)
    assert result["profile"] == profile and result["checkpoint_revision"] == profile + "-revision"
    assert max(reads.index(name) for name in ("config.json", "tokenizer_config.json")) < min(
        reads.index(name) for name in ("model.safetensors", "speech_tokenizer/model.safetensors"))
    # A selected 1.7B config cannot silently consume the other checkpoint's weights.
    def mismatched(path):
        name = str(path.relative_to(checkpoint))
        return "wrong-weight" if name == "model.safetensors" else profiles[profile]["hashes"][name]
    monkeypatch.setattr(loader, "sha256", mismatched)
    with pytest.raises(ValueError, match="model.safetensors"):
        package.preflight(checkpoint, prepared)
