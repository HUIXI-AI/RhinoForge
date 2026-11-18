#!/usr/bin/env bash
# Project Runner supplies read-only dependencies; each job owns a fresh venv.
set -euo pipefail
: "${CI_PROJECT_DIR:?Run through the project CI job}"
: "${CI_JOB_ID:?Run through the project CI job}"
: "${RPU_HOST_PYTHON:?Runner must provide its dedicated Python 3.12}"
: "${RPU_HOST_SITE_PACKAGES:?Runner must provide its dedicated dependency directory}"

cd "$CI_PROJECT_DIR"
rf_job_venv="$CI_PROJECT_DIR/.ci-venv-$CI_JOB_ID"
test ! -e "$rf_job_venv"
"$RPU_HOST_PYTHON" -I -B -m venv "$rf_job_venv"

# Do not inherit developer site-packages or an editable rpu_backend install.
export PYTHONPATH="$CI_PROJECT_DIR:$CI_PROJECT_DIR/python:$RPU_HOST_SITE_PACKAGES"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export PIP_REQUIRE_VIRTUALENV=true PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
# Explicit nodes keep local regression checks proportional to the change.
if (( $# == 0 )); then
    "$rf_job_venv/bin/python" scripts/check_release_version.py
    "$rf_job_venv/bin/python" scripts/check_dependency_release.py
    "$rf_job_venv/bin/python" scripts/check_public_hygiene.py
    "$rf_job_venv/bin/python" scripts/check_examples.py
    set -- tests python/rpu_backend/tests/acceptance python/rpu_backend/tests/core/test_pi05_denoise_buffers.py \
        python/rpu_backend/tests/core/test_pi05_denoise_build_layer_compiles.py \
        python/rpu_backend/tests/core/test_rpu_warmup_binding.py python/rpu_backend/tests/graph \
        python/rpu_backend/tests/library python/rpu_backend/tests/model python/rpu_backend/tests/quant \
        python/rpu_backend/tests/ops/test_siglip_pre_embedded_rows.py
    # Device cases are retained in source and run separately from the host job.
    while IFS= read -r rf_device_test; do
        case "$rf_device_test" in ''|\#*) continue ;; esac
        test -f "$rf_device_test"
        set -- "$@" "--ignore=$rf_device_test"
    done < scripts/ci/device_test_paths.txt
    while IFS= read -r rf_device_node; do
        case "$rf_device_node" in ''|\#*) continue ;; esac
        test -f "${rf_device_node%%::*}"
        set -- "$@" "--deselect=$rf_device_node"
    done < scripts/ci/device_test_nodes.txt
    if (( ${CI_NODE_TOTAL:-1} > 1 )); then
        rf_sharded_out=$("$rf_job_venv/bin/python" - "$@" <<'PY'
import os, pathlib, sys
total = int(os.environ["CI_NODE_TOTAL"])
index = int(os.environ["CI_NODE_INDEX"])
if not (1 <= index <= total):
    raise SystemExit(f"invalid CI_NODE_INDEX={index} for CI_NODE_TOTAL={total}")
roots, ignores, deselects = [], set(), []
for arg in sys.argv[1:]:
    if arg.startswith("--ignore="):
        ignores.add(str(pathlib.Path(arg.split("=", 1)[1]).resolve()))
    elif arg.startswith("--deselect="):
        deselects.append(arg)
    else:
        roots.append(arg)
files = []
for root in roots:
    path = pathlib.Path(root)
    if path.is_file():
        if str(path.resolve()) not in ignores:
            files.append(root)
    else:
        for item in sorted(path.rglob("*.py")):
            if str(item.resolve()) not in ignores and (item.name.startswith("test_") or item.name.endswith("_test.py")):
                files.append(item.as_posix())
selected = files[index - 1::total]
if not selected:
    raise SystemExit(f"empty host-runtime shard {index}/{total}")
for item in (*selected, *deselects):
    print(item)
PY
        )
        mapfile -t rf_sharded_args <<< "$rf_sharded_out"
        set -- "${rf_sharded_args[@]}"
    fi
fi
exec "$rf_job_venv/bin/python" -m pytest -q -ra --durations=20 \
    --junitxml=test-results/host-runtime.xml "$@"
