# 受限运行时资产

简体中文 | [English](runtime_assets.md)

RhinoForge 源码不包含 Rhino Launch 二进制开发包和合并后的 opaque 算子资产。
对于 RhinoForge v1.0.0，获批分发方通过一份不可变交付提供两者：

```text
RhinoForge-runtime-v1.0.0-r4.tar.gz
RhinoForge-runtime-v1.0.0-r4.tar.gz.sha256
```

该归档兼容 RhinoForge v1.0.0 **release**，不包含 RhinoForge 源码、模型、量化权重
或板卡 SDK，且不适用源码仓库的 Apache-2.0 许可证。使用前请阅读
`DISTRIBUTION_TERMS.txt`。

## 交付目录

完成外层归档校验并解包后，目录结构如下：

```text
RhinoForge-runtime-v1.0.0/
├── RELEASE.txt
├── README.txt
├── DISTRIBUTION_TERMS.txt
├── SHA256SUMS
├── <launch-package-file>
├── <opaque-operator-asset>
└── <opaque-operator-asset>.kernels
```

`RELEASE.txt` 是 runtime-set ID、兼容 RhinoForge release、Launch 包文件名/版本、
算子文件名/版本、相邻 manifest 文件名、状态、下载开关和 checksum 范围的唯一来源。
`SHA256SUMS` 覆盖三个运行时 payload：Launch、opaque 算子资产和相邻 manifest。

`.kernels` 文件是分发方根据已审查的公开 host 侧可达性契约生成，并绑定准确资产的
release metadata。它不是算子源文件，也不暴露 opaque payload 内容。必须保持它与
资产相邻且内容不变。

## 1. 获取归档和 checksum

通常由内部交付人员下载并校验两个文件，再把这一对文件转交获批客户；客户收到后重复
校验。不要公开发布，也不要转发给未获批的接收方。

如果分发方提供版本化授权 URL，按下面方式下载同一对文件；否则把收到的两个文件放入
同一本地目录，然后进入下一节。

```bash
set -e
export RHINOFORGE_RUNTIME_BUNDLE_URL="<distributor-provided-runtime-v1.0.0-r4-URL>"
export RHINOFORGE_RUNTIME_ARCHIVE="RhinoForge-runtime-v1.0.0-r4.tar.gz"
curl -fSLO "${RHINOFORGE_RUNTIME_BUNDLE_URL%/}/$RHINOFORGE_RUNTIME_ARCHIVE"
curl -fSLO "${RHINOFORGE_RUNTIME_BUNDLE_URL%/}/$RHINOFORGE_RUNTIME_ARCHIVE.sha256"
```

凭据只能放在源码 checkout 和模型 TOML 之外。使用认证渠道时，请通过独立可信路径
提供的 fingerprint 或证书验证服务器身份。

## 2. 校验并解包交付

在两个文件所在目录执行：

```bash
set -e
export RHINOFORGE_RUNTIME_ARCHIVE="RhinoForge-runtime-v1.0.0-r4.tar.gz"
sha256sum -c "$RHINOFORGE_RUNTIME_ARCHIVE.sha256"
tar -xzf "$RHINOFORGE_RUNTIME_ARCHIVE"
cd RhinoForge-runtime-v1.0.0
cat DISTRIBUTION_TERMS.txt
```

外层 checksum 用于发现传输损坏；获批交付渠道和分发方用于确认来源。来自无关来源的
checksum 不能证明交付可信。

## 3. 校验 release 映射与 payload

只有以下 metadata 精确存在时才继续。所有 payload 文件名必须从 `RELEASE.txt`
读取；不要硬编码或重命名算子文件：

```bash
set -e
grep -qx 'status=active' RELEASE.txt
grep -qx 'download_enabled=true' RELEASE.txt
grep -qx 'compatible_rhinoforge_release=v1.0.0' RELEASE.txt
grep -qx 'checksum_scope=all_runtime_payloads' RELEASE.txt

export RHINOFORGE_RUNTIME_SET="$(sed -n 's/^runtime_set=//p' RELEASE.txt)"
export RHINOFORGE_LAUNCH_PACKAGE="$(sed -n 's/^launch_package_file=//p' RELEASE.txt)"
export RHINOFORGE_OPERATOR_ASSET="$(sed -n 's/^operator_asset_file=//p' RELEASE.txt)"
export RHINOFORGE_OPERATOR_KERNEL_MANIFEST="$(sed -n 's/^operator_kernel_manifest_file=//p' RELEASE.txt)"

case "$RHINOFORGE_RUNTIME_SET" in ""|*/*) echo "invalid runtime-set ID"; exit 1;; esac
case "$RHINOFORGE_LAUNCH_PACKAGE" in ""|*/*) echo "invalid Launch filename"; exit 1;; *.tar.gz) :;; *) echo "invalid Launch filename"; exit 1;; esac
case "$RHINOFORGE_OPERATOR_ASSET" in ""|*/*) echo "invalid operator filename"; exit 1;; *.ref) :;; *) echo "invalid operator filename"; exit 1;; esac
test "$RHINOFORGE_OPERATOR_KERNEL_MANIFEST" = "$RHINOFORGE_OPERATOR_ASSET.kernels"
test -f "$RHINOFORGE_LAUNCH_PACKAGE"
test -f "$RHINOFORGE_OPERATOR_ASSET"
test -f "$RHINOFORGE_OPERATOR_KERNEL_MANIFEST"
sha256sum -c SHA256SUMS
export RHINOFORGE_BUNDLE_DIR="$PWD"
```

`sha256sum` 必须对三个指定运行时 payload 分别报告成功。任何 metadata、文件名或
checksum 失败后都不能继续，也不能混用不同 runtime set 的文件。

## 4. 无 root 权限安装

将 Rhino Launch 解压到根据 manifest 文件名派生的版本化用户目录：

```bash
export RHINO_LAUNCH_ROOT="$HOME/.local/opt/${RHINOFORGE_LAUNCH_PACKAGE%.tar.gz}"
mkdir -p "$HOME/.local/opt"
tar -xzf "$RHINOFORGE_LAUNCH_PACKAGE" -C "$HOME/.local/opt"
test -f "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"

export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

将算子资产和相邻 manifest 一起安装到源码 checkout 和 Python 环境之外：

```bash
export RHINOFORGE_RUNTIME_DIR="$HOME/.local/share/rhinoforge/runtime/$RHINOFORGE_RUNTIME_SET"
install -d "$RHINOFORGE_RUNTIME_DIR"
install -m 0644 "$RHINOFORGE_BUNDLE_DIR/$RHINOFORGE_OPERATOR_ASSET" "$RHINOFORGE_RUNTIME_DIR/"
install -m 0644 "$RHINOFORGE_BUNDLE_DIR/$RHINOFORGE_OPERATOR_KERNEL_MANIFEST" "$RHINOFORGE_RUNTIME_DIR/"

export RPU_KERNEL_LIB_PATH="$RHINOFORGE_RUNTIME_DIR/$RHINOFORGE_OPERATOR_ASSET"
test -r "$RPU_KERNEL_LIB_PATH"
test -r "$RPU_KERNEL_LIB_PATH.kernels"
```

RhinoForge 链接外部 Launch 库，并读取 `RPU_KERNEL_LIB_PATH` 选择的唯一合并资产；
不会把两者复制进 Python 包。保持算子资产 opaque 且内容不变。不要拆分资产、检查其
内容、自行生成替代 manifest 或配置其他算子库路径。

普通源码安装、验证、精确 Qwen3-0.6B 下载、TOML 检查和推理步骤见
[快速开始](getting_started.zh.md#4-在全新环境中安装-rhinoforge)。

## 更新或撤销交付

已经发布的 runtime set 不可变。替代版本必须安装到新的 runtime-set 目录，不能覆盖
原集合。分发方撤销某一交付后，应停止使用并删除该版本，只迁移到分发方指定的替代
版本。RhinoForge 不会自动下载、更新或撤销受限资产。
