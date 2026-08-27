# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities through
[GitHub private vulnerability reporting](https://github.com/HUIXI-AI/RhinoForge/security/advisories/new).
Include the affected RhinoForge version, a minimal reproduction, impact, and
any relevant Rhino Launch and operator-asset versions or hashes.

Do not report vulnerabilities in a public issue, discussion, or pull request.
Do not attach restricted binaries, operator assets, model checkpoints, secrets,
personal data, or private infrastructure details. Describe how an authorized
maintainer can reproduce the issue instead.

Maintainers will acknowledge the private report, investigate the supported
release scope, coordinate remediation and disclosure when appropriate, and
publish an advisory for confirmed issues that affect a public release.

## Scope

This policy covers RhinoForge source and release artifacts published by the
HUIXI-AI organization. Separately distributed Rhino Launch packages, operator
assets, model checkpoints, third-party dependencies, and components of the
preconfigured board environment remain subject to their owners' security
processes. A report that appears to belong elsewhere may be redirected without
exposing its contents publicly.

Only versions identified as supported in a published release are eligible for
security fixes. Source candidates and experimental model profiles may be used
to reproduce an issue but do not create a support commitment.

## Trusted inputs

Generic Hugging Face loaders and Pi0.5 do not execute remote model code.
RhinoVLA `runtime_factory` values are executable installed integrations and
must come from reviewed configuration. Hy-Embodied `norm_stats.pkl` files are
also executable inputs: the required SHA-256 pins reviewed bytes but does not
make an untrusted pickle safe.

Treat model checkpoints, tokenizers, processors, configuration, and
pickle-backed statistics as trusted inputs. Review their source, revision,
license, digest, and configuration before loading them.

RhinoForge declares `torch==2.10.0`. PyTorch security advisories apply when
they affect that version and a reachable deployment code path; consult the
upstream advisory and platform-provider guidance. RhinoForge does not claim to
remediate vulnerabilities in PyTorch or another upstream dependency.

The v1.0.0 source does not call `torch.jit.script` or load `.pt2` packages, so
[PYSEC-2025-194](https://github.com/advisories/GHSA-rrmf-rvhw-rf47) and
[PYSEC-2026-139](https://osv.dev/vulnerability/PYSEC-2026-139) are not
reachable through RhinoForge's tracked code. Applications that use those
interfaces remain responsible for the upstream risk. Reassess the dependency
before adding either interface.

Privileged board-device access is intended only for trusted users on a
single-tenant host. Do not expose it to untrusted code or mutually untrusted
tenants.
