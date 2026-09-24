"""Public package checks independent of model qualification records."""
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_default_install_includes_model_loading_dependency():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "rhinoforge"
    assert any(dependency.startswith("accelerate") for dependency in project["dependencies"])


def test_kernel_manifest_uses_host_symbols_without_reading_asset_format(tmp_path):
    asset = tmp_path / "operator.ref"
    asset.write_bytes(b"opaque")
    manifest = Path(str(asset) + ".kernels")
    available = tmp_path / "vendor-kernels.txt"
    available.write_text("unary_cast_uint8_fp16\nunreachable_test_kernel\n", encoding="ascii")
    subprocess.run(
        [sys.executable, str(ROOT / "release/generate_kernel_manifest.py"),
         str(asset), str(manifest), "--available-kernels", str(available)], check=True,
    )
    rows = manifest.read_text(encoding="ascii").splitlines()
    assert rows[:2] == ["rhinoforge-kernels-v1", "asset-size=6"]
    assert rows[2:] == ["unary_cast_uint8_fp16"]
    assert asset.read_bytes() == b"opaque"
