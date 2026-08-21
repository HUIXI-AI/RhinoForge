# Runtime profiles

A RhinoForge runtime profile is the complete execution contract for one model
configuration. It is not a loose collection of environment switches.

## Profile identity

Treat a profile as a tuple of:

- model/checkpoint revision and precision or quantization format;
- admitted input, sequence, image/view, batch, and action envelope;
- immutable per-handle execution planning;
- RhinoForge, Rhino Launch, and combined operator-asset compatibility set; and
- effective public runtime configuration.

Changing any member creates a new profile that needs its own admission,
correctness, lifecycle, and performance evidence. An architecture-name match or
registry alias does not inherit another profile's support status.

## Configuration layers

The full TOML runner applies allowlisted `[runner.env]` values before importing
the backend. The deployment environment supplies restricted asset locations and
other deployment-owned values. Model facades may then provide profile-owned
defaults without overwriting explicit values. Native and model state can bind
settings at import, first native use, model construction, or Graph BUILD.

Therefore:

- start a fresh process when changing an import-, native-, model-, or
  build-bound value;
- prefer facade or loader configuration over exporting model-specific switches
  globally;
- record effective values, not only the TOML text; and
- never infer that changing `os.environ` reconfigured an existing model or
  Graph.

## Safe profile changes

Classify a setting before changing it:

| Class | Typical effect | Required response |
|---|---|---|
| Deployment binding | Selects verified runtime/model assets | Re-verify compatibility and provenance |
| Execution planning | Changes chunks, padding, memory, or Graph signatures | Rebuild the model/Graphs and rerun envelope gates |
| Numerical profile | Changes precision, accumulation, quantization, or model inputs | Treat as a new correctness profile |
| Performance topology | Changes batching, replay, fusion, or scheduling | Prove semantic equivalence and lifecycle before timing |
| Diagnostic | Adds logs, exports, traces, or synthetic overrides | Keep out of production and review generated artifacts |

Do not compose a production profile by copying individual fast-path switches
from another model. Use the owning facade's complete admitted profile and
override only settings documented for that exact path.

## Sources

- [Runtime configuration](../../docs/runtime_config.md)
- [Model support](../../docs/model_support.md)
- [Model execution and profiling](../../docs/model_testing.md)
- [Performance measurement](../../docs/performance.md)
- [Restricted runtime assets](../../docs/runtime_assets.md)
