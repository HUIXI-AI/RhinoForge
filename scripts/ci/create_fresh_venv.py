"""Reuse base dependencies without executing its editable-install startup hooks."""
from pathlib import Path
import sys
import sysconfig
import venv


def create(target: Path, dependency_site: Path | None = None) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"fresh environment already exists: {target}")
    if dependency_site is None:
        # Keep upgrades installed in the active pipeline overlay ahead of the
        # immutable torch base, without executing either layer's .pth hooks.
        dependency_sites = [Path(sysconfig.get_path("purelib")), Path(sysconfig.get_path(
            "purelib", vars={"base": sys.base_prefix, "platbase": sys.base_prefix}))]
    else:
        dependency_sites = [dependency_site]
    dependency_sites = list(dict.fromkeys(path.resolve() for path in dependency_sites))
    for path in dependency_sites:
        if not path.is_dir():
            raise FileNotFoundError(path)
    venv.EnvBuilder(with_pip=True, symlinks=True).create(target)
    site = Path(sysconfig.get_path(
        "purelib", vars={"base": str(target), "platbase": str(target)}))
    # A plain path adds dependencies after this venv's installed packages.
    # Unlike --system-site-packages / addsitedir, it does not run base .pth files.
    (site / "rhinoforge_ci_dependencies.pth").write_text(
        "".join(str(path) + "\n" for path in dependency_sites))


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("fresh installation checks require Python 3.12")
    create(Path(sys.argv[1]).absolute())
