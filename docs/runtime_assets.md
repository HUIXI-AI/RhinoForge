# Restricted runtime assets

RhinoForge source does not include the two restricted runtime prerequisites:

- the Rhino Launch binary development package; and
- one combined operator asset named `rhinoOpLib_current.ref`.

Obtain both from the same authorized runtime release. Do not combine a Launch
package, operator asset, or BOM from different releases.

## Authorized release index

Access details are supplied after authorization and are not stored in this
repository. Customer and partner delivery must use either HTTPS with an
authorized account or scoped read-only credential, or SFTP/SSH with an
authorized key. Never put a token in a URL, repository, TOML file, shell
history, or support report.

The distribution service may use a private GitLab Release backed by its Generic
Package Registry, an HTTPS static artifact service, SFTP, or an equivalent
controlled service. A web page or an SFTP directory is only the index; the
versioned directory holds the files. This choice does not add a build-time or
runtime dependency to RhinoForge.

A plain HTTP index is acceptable only for a temporary engineering preview that
is bound to a network-restricted internal interface, does not accept or expose
credentials, and clearly marks its artifacts as internal-only. It is not a
customer distribution channel. Do not send an account, password, bearer token,
or cookie over plain HTTP.

Each index represents one immutable compatibility set and must show:

- compatibility-set and BOM IDs;
- `active`, `superseded`, or `revoked` status and its validity period;
- compatible RhinoForge release, Rhino Launch version, host platform, and
  preconfigured board-environment compatibility identifier;
- combined operator-asset filename, size, and SHA256;
- license, access, support, and replacement information; and
- the organization signing-key fingerprint and canonical status location.

Select the exact compatibility set named by the RhinoForge release. Do not use
an unversioned `latest` link for installation.

## Files to download

The index must provide these files or links for the selected set:

| File | Purpose |
|---|---|
| Rhino Launch development-package archive | Headers, shared library, CMake package files, and its own license files |
| Launch checksum and install manifest | Verify the archive and its installed file set |
| `rhinoOpLib_current.ref` | The single combined operator asset used at runtime |
| `rhinoOpLib_current.ref.sha256` | Verify the downloaded operator asset |
| `asset-compat-bom.v1.json` and detached signature | Bind the Launch package, operator asset, required runtime surface, and compatibility identifiers |
| `release-bom.v1.json` and detached signature | Bind the RhinoForge source release to the compatible asset BOM |
| `release-status.v1.json` and detached signature | State whether the exact BOM and artifacts are current, superseded, or revoked |

The Launch archive must also include `LICENSE` and `NOTICE`, when applicable.
Their terms are separate from RhinoForge's Apache-2.0 license.

## Download and verify

1. Open the authorized release index supplied by the distributor and select the
   compatibility set named by the RhinoForge release. For SFTP, connect with
   the supplied host, account, and SSH key, then enter that versioned directory.
2. For customer or partner delivery, confirm that the web index uses HTTPS or
   that the SFTP host key matches the independently supplied fingerprint.
   Confirm that the set is not marked `revoked`. A plain HTTP index is valid
   only when it is explicitly labelled as an internal engineering preview.
3. Download the three JSON records, their detached signatures, and the
   organization trust-root information first.
4. Run the exact signature-verification command published in the index. Verify
   the trust-root fingerprint through the distributor's independent support or
   security channel. Do not continue unless all three signatures pass.
5. Confirm that `release-status.v1.json` is current, unexpired, and names the
   same release-BOM and asset-BOM IDs. `active` is the normal installation
   state. A `superseded` set may be used only when its signed status explicitly
   remains supported.
6. Confirm that `release-bom.v1.json` names the RhinoForge version being built
   and references the downloaded `asset-compat-bom.v1.json`.
7. Download the Launch archive, its checksum and install manifest, the combined
   operator asset, and its checksum.
8. In the download directory, verify both artifacts:

   ```bash
   sha256sum -c RHINO_LAUNCH_ARCHIVE.sha256
   sha256sum -c rhinoOpLib_current.ref.sha256
   ```

   Replace `RHINO_LAUNCH_ARCHIVE.sha256` with the exact checksum filename shown
   in the release index. The resulting hashes must also equal the values in the
   verified asset BOM.

Stop if a signature, status, identifier, checksum, license, or compatibility
check differs. Do not work around a mismatch by renaming or combining files.

## Install Rhino Launch

Accept the terms delivered with the Launch package, then follow the exact
archive layout and installation command shown in the authorized release index.
Use a versioned installation prefix and confirm that it contains the headers,
versioned shared library, and CMake package files recorded by the install
manifest.

For a non-system installation, expose that exact prefix to CMake:

```bash
export RHINO_LAUNCH_PREFIX="/path/to/rhino-launch/VERSION"
test -f "$RHINO_LAUNCH_PREFIX/lib/cmake/rhino_launch/rhino_launchConfig.cmake"
export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
```

The shared library must also be visible through the board environment's normal
system-loader configuration. RhinoForge does not copy it into the Python
package.

## Install the combined operator asset

Keep the verified asset outside the source checkout and Python environment.
Install it under the compatibility-set ID so different sets cannot overwrite
one another:

```bash
export RHINOFORGE_RUNTIME_SET="COMPATIBILITY_SET_ID"
export RHINOFORGE_RUNTIME_DIR="$HOME/.local/share/rhinoforge/runtime/$RHINOFORGE_RUNTIME_SET"
install -d "$RHINOFORGE_RUNTIME_DIR"
install -m 0644 rhinoOpLib_current.ref "$RHINOFORGE_RUNTIME_DIR/"
export RPU_KERNEL_LIB_PATH="$RHINOFORGE_RUNTIME_DIR/rhinoOpLib_current.ref"
test -r "$RPU_KERNEL_LIB_PATH"
```

Replace `COMPATIBILITY_SET_ID` with the value from the verified asset BOM.
`RPU_KERNEL_LIB_PATH` must point to this single file and must be set in the
deployment environment, not in a model TOML file.

Continue with [Getting started](getting_started.md#4-build-and-install-rhinoforge).

## Superseded and revoked releases

The signed canonical status is authoritative; a badge or edited index alone is
not. A release index and its tombstone remain visible after revocation, but its
restricted binaries must no longer be downloadable. Never overwrite files in
an existing compatibility set: publish a new set and signed status instead.

If a set is `revoked`, stop using it, remove its local artifacts according to
your organization's policy, and install the named replacement. RhinoForge does
not contact the distribution service or update restricted assets automatically.
