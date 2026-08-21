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
