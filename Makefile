PYTHON ?= python3
PYTEST_ARGS ?=
WHEEL_DIR ?=
VENV_DIR ?=

.PHONY: ci-local-core ci-local-quickstart ci-local-fresh-venv ci-local-smoke help

help:
	@echo "RhinoForge local checks:"
	@echo "  ci-local-core        host regression tests"
	@echo "  ci-local-quickstart  example configs and adapter registry"
	@echo "  ci-local-fresh-venv  build and install a wheel in a new environment"
	@echo "  ci-local-smoke       quickstart and wheel installation"
	@echo "Override PYTHON; WHEEL_DIR and VENV_DIR must be new paths."

ci-local-core:
	@set -eu; \
	set -- tests python/rpu_backend/tests/acceptance \
		python/rpu_backend/tests/core/test_pi05_denoise_buffers.py \
		python/rpu_backend/tests/core/test_pi05_denoise_build_layer_compiles.py \
		python/rpu_backend/tests/core/test_rpu_warmup_binding.py \
		python/rpu_backend/tests/graph python/rpu_backend/tests/library \
		python/rpu_backend/tests/model python/rpu_backend/tests/quant \
		python/rpu_backend/tests/ops/test_siglip_pre_embedded_rows.py; \
	while IFS= read -r path; do \
		case "$$path" in ''|\#*) continue ;; esac; \
		set -- "$$@" --ignore="$$path"; \
	done < scripts/ci/device_test_paths.txt; \
	while IFS= read -r node; do \
		case "$$node" in ''|\#*) continue ;; esac; \
		set -- "$$@" --deselect="$$node"; \
	done < scripts/ci/device_test_nodes.txt; \
	PYTHONPATH=.:python TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
		$(PYTHON) -m pytest -q "$$@" $(PYTEST_ARGS)

ci-local-quickstart:
	$(PYTHON) -m examples.run --config examples/configs/qwen3/text/0_6b/fp16.toml --check-config
	$(PYTHON) -m examples.run --config examples/configs/pi05/h50/w8a16.toml --check-config
	$(PYTHON) scripts/check_examples.py
	$(PYTHON) scripts/check_supported_models_table.py

ci-local-fresh-venv:
	@set -eu; \
	work=$$(mktemp -d "$${TMPDIR:-/tmp}/rhinoforge-local.XXXXXX"); \
	trap 'rm -rf -- "$$work"' EXIT HUP INT TERM; \
	wheel_dir="$(WHEEL_DIR)"; venv_dir="$(VENV_DIR)"; \
	wheel_dir=$${wheel_dir:-$$work/wheels}; venv_dir=$${venv_dir:-$$work/venv}; \
	for path in "$$wheel_dir" "$$venv_dir"; do \
		if [ -e "$$path" ] || [ -L "$$path" ]; then \
			echo "Output path already exists: $$path" >&2; exit 1; \
		fi; \
	done; \
	test "$$wheel_dir" != "$$venv_dir"; \
	mkdir -p -- "$$wheel_dir"; \
	$(PYTHON) -m pip wheel --no-deps --no-build-isolation -w "$$wheel_dir" .; \
	unset PYTHONPATH PYTHONHOME; \
	$(PYTHON) -I -m venv "$$venv_dir"; \
	"$$venv_dir/bin/python" -I -m pip install "$$wheel_dir"/rhinoforge-*.whl; \
	TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$$venv_dir/bin/python" -I -c "import importlib.metadata, pathlib; import rpu_backend; want = pathlib.Path('VERSION').read_text().strip(); got = importlib.metadata.version('rhinoforge'); assert got == want, (got, want); print('Installed RhinoForge', got)"

ci-local-smoke: ci-local-quickstart ci-local-fresh-venv
