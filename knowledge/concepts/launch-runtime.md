# Host launch runtime boundary

RhinoForge uses the separately distributed Rhino Launch development package as
a link-time library dependency. Model code calls public host wrappers; the
restricted library and combined operator asset remain outside the source tree.
[Restricted runtime assets](../../docs/runtime_assets.md)

## Dependency and compatibility contract

- CMake resolves one exact `rhino_launch` package version. Install its headers,
  shared library, and CMake package files from the same runtime set as the
  combined operator asset.
  [Build dependency](../../CMakeLists.txt)
- The shared library is discovered by the board environment's normal loader;
  RhinoForge does not vendor or copy it into the Python package.
  [Launch installation](../../docs/runtime_assets.md#4-install-without-root)
- Treat the Launch package, operator asset, and RhinoForge release as one
  versioned runtime set. A successful link does not prove runtime compatibility.
  [Runtime set](../../docs/runtime_assets.md#delivery-layout)
- Require the release-provided `<operator-asset>.kernels` sidecar before loading
  any program. It lists the asset size and names only; RhinoForge never parses
  the opaque asset.
  [Kernel manifest](../../docs/runtime_assets.md#delivery-layout)

## Public wrapper contract

The reviewable boundary is the host wrapper: tensor shapes and dtypes, memory
ownership, addresses or offsets, transfer length, participating-core selection,
synchronization, and parameter packing may be represented in source. Restricted
device-program implementation and operator-asset internals are not part of this
knowledge base. [Operator boundary](../../docs/architecture.md#operators-and-cpu-fallback)

Keep these ownership rules explicit when adding a wrapper:

- ordinary operators use the address form declared by their wrapper; do not
  interchange an absolute SPM address and an offset;
- immediate multi-core work enables broadcast for all participating cores;
- fixed, mutable, and immediate DMA wrappers follow their distinct address-
  lifetime contracts; and
- wrapper success is only an execution result. Model support still requires
  numerical, cache, and Graph lifecycle gates.

See [SPM allocation](spm-allocation.md),
[multi-core broadcast](multicore-broadcast.md), and
[DMA wrappers](ddr-dma-wrappers.md).

## Queue and batch ownership

Graph BUILD groups operator, DMA, and synchronization nodes into launch
segments. Every retained Graph entry owns its prepared launch queue. A bounded
one-shot execution uses a one-shot queue path, while a direct passthrough launch
uses an isolated direct-launch path. Do not interleave direct launch work with a
prepared retained batch on the same queue.
[Graph execution](../../src/graph/graph_runtime_execute.cpp)

Per-node core selection is preserved inside a batch, so one-core and multi-core
nodes may coexist. Mutable DMA parameters are refreshed before replay; a change
that affects graph structure or layout requires a new signature or model-state
invalidation instead of a parameter patch.
[Graph and batch execution](../../docs/architecture.md#graph-and-batch-execution)

## Sources

- [CMake dependency](../../CMakeLists.txt)
- [Architecture](../../docs/architecture.md)
- [Restricted runtime assets](../../docs/runtime_assets.md)
- [Graph runtime](../../src/graph/graph_runtime_execute.cpp)
- [Public DMA declarations](../../src/core/rpu_kernel_decls.h)
- [Model execution and profiling](../../docs/model_testing.md)
