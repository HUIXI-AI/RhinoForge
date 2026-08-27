# Lazy adapter registry

Importing `rpu_backend` registers built-in architecture names without importing
every model implementation. The first lookup of an architecture loads only its
declared adapter. This keeps optional model dependencies and model-specific
class patches out of the base import path.

## Ownership

- `python/rpu_backend/adapters/_manifest.py` is the data-only built-in map.
- `python/rpu_backend/runtime/registry.py` owns registration and lookup without
  model-family knowledge.
- Each adapter registers the same architecture/class pair when imported, so a
  direct import and a lazy lookup converge on one binding.
- External `rpu_backend.plugins` entry points are considered only after an
  unknown built-in lookup or an explicit discovery request.
- A plugin may add a new name but may not replace a built-in or previously
  registered binding.

Loading a plugin executes installed Python code. Enable only reviewed plugin
distributions, and use a fresh process after changing the installed plugin set.

## Adding an adapter

Update the manifest and module registration together, keep module import free
of model loading or device mutation, and add an early configuration-only route
check. Do not restore package-wide module scanning: it defeats lazy imports and
makes unrelated optional dependencies affect the base package.

## Sources

- [Adapter registry and plugins](../../docs/api_reference.md#adapter-registry-and-plugins)
- [Built-in manifest](../../python/rpu_backend/adapters/_manifest.py)
- [Registry implementation](../../python/rpu_backend/runtime/registry.py)
- [Model porting](../../docs/model_porting.md#6-register-the-adapter)
