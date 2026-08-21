# Contributing to RhinoForge

Contributions should be small, reviewable, and tied to a concrete inference or
model-porting need. Open an issue before beginning a large model port or a new
fused subsystem so the intended profile and public operator coverage can be
agreed first.

## Development setup

Use a preconfigured RPU board environment and install the exact Rhino Launch
1.0.0 development package. If its CMake package is outside the system search
path, add its prefix to `CMAKE_PREFIX_PATH`, then install from the repository
root:

```bash
python -m pip install -e '.[dev]' --no-build-isolation
```

Set `RPU_KERNEL_LIB_PATH` to the release-matched combined operator asset for
board execution. Restricted packages and assets must stay outside the
repository.

## Change requirements

- Keep the Python import name `rpu_backend` and the public interfaces documented
  in [`docs/api_reference.md`](docs/api_reference.md).
- Treat FP16, SPM capacity, multi-core dispatch, graph lifetime, and irreversible
  weight swizzling as explicit contracts rather than inferred behavior.
- Reject unsupported model profiles before downloading or mutating weights.
- Reuse public operators and the adapter template before adding a fused C++
  subsystem. Follow [`docs/model_porting.md`](docs/model_porting.md).
- Add the smallest board-free test that protects new validation or planning
  logic. Model execution changes also require CPU-reference versus RPU evidence
  on the affected profile.
- Update the model, asset, API, runtime-configuration, or quantization document
  whenever a public contract changes.

Do not commit device assembly, ISA or opcode semantics, instruction streams,
operator-asset format parsers, disassemblers, generator sources, restricted
binaries or assets, credentials, private paths, private infrastructure details,
or customer-confidential material. Host-side operator wrappers, SPM planning,
multi-core scheduling, graph/batch execution, DMA behavior, and launch ABI
integration are in scope.

## Validation

Run the checks applicable to the change:

```bash
python -m compileall -q python/rpu_backend examples
python -m pytest
python -m build --sdist --wheel
```

Hardware changes additionally need the profile-specific numerical,
warmup/replay, graph-lifecycle, and multi-input checks described in
[`docs/model_porting.md`](docs/model_porting.md). Record the exact source,
Rhino Launch, operator-asset, model-checkpoint, and configuration versions used.

Before submitting, inspect the complete diff for generated files, private
information, restricted artifacts, and unrelated cleanup. Contributions are
accepted under the repository's [Apache-2.0 license](LICENSE).

## Sign your work

RhinoForge uses the [Developer Certificate of Origin 1.1](DCO.md). Add a
`Signed-off-by` line to every commit with Git's standard sign-off option:

```bash
git commit -s
```

The sign-off certifies that you have the right to submit the contribution under
the project license. Unless explicitly stated otherwise, intentionally
submitted contributions are licensed under Apache-2.0 as described by section
5 of that license. Public pull requests pass through the project's private
release and hardware validation gate before they are eligible to merge.
