# Version and release policy

RhinoForge's root [VERSION](../VERSION) supplies its Python metadata and numeric
CMake version. Rhino Launch is an independently versioned dependency pinned in
[runtime-dependencies.json](../release/runtime-dependencies.json). Advancing
RhinoForge's version does not advance Launch or rename a published runtime bundle.
Package names retain their full version, including any development suffix.

| Line | Version | Purpose |
|---|---|---|
| `main` | `1.3.0.dev0` | Next release development |
| `release/1.2` | `1.2.0` | Current stable release |
| `release/1.2` after release | `1.2.1` | Compatible fixes |
| Historical tags | Original version | Reproducible source and downloads |

Cut the release branch before advancing `main`. Launch uses `develop` for its
own development line. RhinoForge `main` currently retains archived Launch
`1.2.0` at `3dd1b410f5f46373997fef3dfbf9b67eecec6a84`, ABI `1`.
Change that dependency only when adopting a new Launch revision, with the matching
package and affected compatibility checks. Merge release fixes back into
development while keeping each project's version metadata.

Tag each stable release once, such as `v1.2.0`. Its Release records the date,
source and component commits, and package checksums; a duplicate `archive/` tag
is unnecessary. The current download entry is `1.2.0`. Keep superseded candidate
packages and their original checksums in the download history. Before removing
a candidate tag or Release, migrate its records and update download links.
See [source history](release_history.md) for retired source references.

Keep stable tags and published package bytes fixed. Advance `main` independently;
compatible stable fixes receive a new patch version. Retain `main` and the active
release branches, and remove merged working branches after checking for new
commits. A development entry may provide source only.

## Build and dependency versions

[runtime-dependencies.json](../release/runtime-dependencies.json) records
`op_abi_epoch` alongside the Launch source commit, full `release_version`,
numeric `cmake_version` and `abi_version`. The legacy `version` field equals
`cmake_version`, which is the numeric base of Launch's own `release_version`.
CMake checks the installed Launch package against these exact dependency versions
and ABI, and retains capability checks. Launch's version need not equal
RhinoForge's version. Library SONAME and Python `API_VERSION` track
compatibility independently of distribution versions.

Runtime contains exactly the main and DDR-expansion operator assets pinned by
commit and checksum, governed by `op_abi_epoch` (currently `1`, incremented only
on an incompatible operator payload ABI break). Their component source versions
need not match the bundle version. Keep each payload and its `.kernels` manifest
together.

Run `python scripts/check_release_version.py` before packaging; CI also checks
the branch or tag against `VERSION`. Validate Python metadata, the selected Launch
package and the affected build or runtime path before publishing. Keep model
evaluation records outside release artifacts.

## Dependency environment releases

The [dependency bundle design](environment_bundle.md) separates public third-party
environments from RhinoForge source, platform SDK libraries and restricted runtime
assets. Environment versions and immutable archives are independent of RF versions;
source checkouts bind to a compatible environment by ID, locks and checksums.
The first environment is the `deps-2026.10.1` preview; its actual archive and
checksums are bound by [environment.json](../release/environment.json). `deps-*` tags
use `check_dependency_release.py`, dispatched by `check_release_version.py`;
the existing `v*`/`VERSION` source tag check remains unchanged. Do not tag or upload a legacy full bundle as a public
dependency release.

## 中文说明

RhinoForge 的 `VERSION` 决定自身发行版本，Rhino Launch 独立版本化并由依赖清单固定。
`release/1.2` 发布正式版 `1.2.0`，`main` 开发 `1.3.0.dev0`，继续依赖已归档的
Launch `1.2.0`（`3dd1b410f5f46373997fef3dfbf9b67eecec6a84`，ABI `1`）。
RhinoForge 升版不要求 Launch 同步升版，也不改名或覆盖已发布的 runtime 包。
Launch 自身的开发分支为 `develop`；采用新 Launch 提交时，单独更新依赖、获取配套包，
并验证受影响的兼容性。兼容修复递增各自的补丁号，并合回开发分支。

每个正式版只保留一个标准标签，如 `v1.2.0`；Release 记录日期、源码及组件 commit
和包校验和，无需重复创建 `archive/` 标签。当前下载入口为 `1.2.0`。被替代的候选包
及原校验和保留在下载历史中，删除候选标签或 Release 前先迁移记录并更新链接。

正式标签和已发布包保持固定。`main` 独立开发，兼容修复递增补丁号。长期保留
`main` 和仍在维护的 release 分支；工作分支合并后确认没有新增提交即可删除。
开发版可仅提供源码。

CMake 数值版本不含开发后缀。Launch 的数值版本、完整发行版本和 ABI 按依赖清单
精确核对，并保留能力检查；它们不要求与 RhinoForge 同号。Launch ABI 和 Python
API 兼容版本独立维护。依赖清单同时通过 `op_abi_epoch` 标记算子载荷 ABI 纪元（当前为
`1`，仅在不兼容算子 ABI 变更时递增）；算子资产固定来自 main 与 DDR-expansion 两个分支，
并记录准确 commit 和校验和。发布前检查版本、包装和受影响的构建或运行路径；
模型评测记录不进入发行资产。

公开依赖环境包采用独立版本，源码以环境 ID、依赖锁和校验和关联；不携带 RF、
板卡 SDK 库或受限运行资产。首个环境为 `deps-2026.10.1` 预览版，
实际下载与摘要由 [environment.json](../release/environment.json) 固定，见[依赖环境包说明](environment_bundle.zh.md)。
`deps-*` 环境标签由 `check_dependency_release.py` 独立检查，现有 `v*` 源码标签规则保持不变。
