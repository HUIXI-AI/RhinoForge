# 快速开始

简体中文 | [English](getting_started.md)

这是 RhinoForge v1.0.0 的客户 Quick Start。流程从 RhinoForge v1.0.0
源码 release 和获批交付包 `RhinoForge-runtime-v1.0.0-r4.tar.gz` 开始，最终使用
仓库内 TOML 完成 Qwen3-0.6B 推理。板卡 SDK/runtime 和设备权限必须预先配置完成；
交付包不包含模型。

请在同一个 shell 中执行全部命令，以保留导出的运行时路径。

## 1. 检查板卡环境

使用平台提供的 Python 3.12。第 4 步会在全新虚拟环境中安装 PyTorch 2.10.0；
构建还需要支持 C++17 的编译器。

```bash
python3.12 --version
c++ --version
```

从公开 v1.0.0 release 获取 RhinoForge 源码。运行时兼容关系只由该 release
版本确定。

## 2. 接收并校验运行时交付包

通常由内部交付人员完成校验，再把归档及其相邻 checksum 文件转交获批客户。将两个
文件放在同一目录：

```text
RhinoForge-runtime-v1.0.0-r4.tar.gz
RhinoForge-runtime-v1.0.0-r4.tar.gz.sha256
```

如果分发方提供了授权下载 URL，将 `RHINOFORGE_RUNTIME_BUNDLE_URL` 设置为该版本化
目录，并下载同一对文件：

```bash
set -e
export RHINOFORGE_RUNTIME_BUNDLE_URL="<distributor-provided-runtime-v1.0.0-r4-URL>"
export RHINOFORGE_RUNTIME_ARCHIVE="RhinoForge-runtime-v1.0.0-r4.tar.gz"
curl -fSLO "${RHINOFORGE_RUNTIME_BUNDLE_URL%/}/$RHINOFORGE_RUNTIME_ARCHIVE"
curl -fSLO "${RHINOFORGE_RUNTIME_BUNDLE_URL%/}/$RHINOFORGE_RUNTIME_ARCHIVE.sha256"
```

在收到文件的目录中校验外层归档、解包、阅读分发条款并进入解包目录：

```bash
set -e
export RHINOFORGE_RUNTIME_ARCHIVE="RhinoForge-runtime-v1.0.0-r4.tar.gz"
sha256sum -c "$RHINOFORGE_RUNTIME_ARCHIVE.sha256"
tar -xzf "$RHINOFORGE_RUNTIME_ARCHIVE"
cd RhinoForge-runtime-v1.0.0
cat DISTRIBUTION_TERMS.txt
```

只有交付包状态为 active、允许下载、兼容 RhinoForge v1.0.0 release，且 checksum
覆盖全部三个运行时 payload 时才继续。所有 payload 文件名都从 `RELEASE.txt`
读取；不要硬编码或重命名 opaque 算子文件：

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

最后一条校验命令必须分别报告 Launch 包、opaque 算子资产及其相邻 `.kernels`
manifest 校验成功。分发与撤销规则见[受限运行时资产](runtime_assets.zh.md)。

## 3. 无 root 权限安装运行时资产

将 Rhino Launch 安装到带版本号的用户目录。目录名从 `RELEASE.txt` 中的包名派生：

```bash
export RHINO_LAUNCH_ROOT="$HOME/.local/opt/${RHINOFORGE_LAUNCH_PACKAGE%.tar.gz}"
mkdir -p "$HOME/.local/opt"
tar -xzf "$RHINOFORGE_LAUNCH_PACKAGE" -C "$HOME/.local/opt"
test -f "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"

export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

将算子资产及其 manifest 一起安装到源码目录和 Python 环境之外：

```bash
export RHINOFORGE_RUNTIME_DIR="$HOME/.local/share/rhinoforge/runtime/$RHINOFORGE_RUNTIME_SET"
install -d "$RHINOFORGE_RUNTIME_DIR"
install -m 0644 "$RHINOFORGE_BUNDLE_DIR/$RHINOFORGE_OPERATOR_ASSET" "$RHINOFORGE_RUNTIME_DIR/"
install -m 0644 "$RHINOFORGE_BUNDLE_DIR/$RHINOFORGE_OPERATOR_KERNEL_MANIFEST" "$RHINOFORGE_RUNTIME_DIR/"

export RPU_KERNEL_LIB_PATH="$RHINOFORGE_RUNTIME_DIR/$RHINOFORGE_OPERATOR_ASSET"
test -r "$RPU_KERNEL_LIB_PATH"
test -r "$RPU_KERNEL_LIB_PATH.kernels"
```

保持资产 opaque 且内容不变。不要拆分资产、检查其内容、自行生成替代 manifest，
也不要把资产路径或下载凭据写入模型 TOML。

## 4. 在全新环境中安装 RhinoForge

将 `RHINOFORGE_SOURCE` 指向 RhinoForge v1.0.0 源码目录。使用真正隔离的全新虚拟
环境，避免与旧的 `rpu_backend` 发行包冲突；不要使用 `--system-site-packages`。

```bash
export RHINOFORGE_SOURCE="/absolute/path/to/RhinoForge"
cd "$RHINOFORGE_SOURCE"
python3.12 -m venv .venv
. .venv/bin/activate

python -m pip install "torch==2.10.0" "scikit-build-core>=0.12,<0.13" "cmake==4.1.3" "ninja==1.13.0"
python -m pip install . --no-build-isolation
python -m pip check
```

第二条命令是普通、非 editable 安装。不要添加 `--no-deps`：默认安装必须自动解析
包括 Accelerate 在内的全部运行依赖。editable 安装只用于源码开发，不属于客户主路径。

检查已安装 release 和模型加载依赖：

```bash
python - <<'PY'
from importlib.metadata import version
from packaging.version import Version

assert version("rhinoforge") == "1.0.0"
assert Version(version("torch")).base_version == "2.10.0"
assert version("transformers") == "5.5.0"
assert Version("1.1.0") <= Version(version("accelerate")) < Version("2")
print("RhinoForge Python dependencies: OK")
PY
```

## 5. 验证安装与 RPU

先检查 import 和版本路径但不执行 device copy，再执行 RPU 数据复制检查：

```bash
python examples/verify_install.py --check-config
python examples/verify_install.py
```

两条命令都必须以零状态码结束。如果第二条命令报告设备权限错误，请让板卡管理员授予
RPU 设备访问权限；不要为此使用 root 重新安装运行时资产。

## 6. 准备 Qwen3-0.6B

v1.0.0 公开模型台账将本 Quick Start profile 固定为 `Qwen/Qwen3-0.6B`
revision `c1899de289a04d12100db370d81485cdf75e47ca`。把它下载到仓库内 registry alias
预期的目录：

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
mkdir -p "$RPU_MODEL_CACHE"
hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-0.6B"
test -f "$RPU_MODEL_CACHE/Qwen3-0.6B/config.json"
```

如果分发方提供了同一精确 checkpoint 的已校验副本，也可以把它放到同一最终目录。
模型许可证和访问条款独立于 RhinoForge 与 Runtime Bundle 条款。

## 7. 检查 TOML 配置

复制仓库内的完整配置。保留其中的 `qwen3-0.6b` alias；它会通过
`RPU_MODEL_CACHE` 解析，因此不需要改写绝对 checkpoint 路径。

```bash
cp examples/configs/qwen3_0_6b.toml qwen3.local.toml
python examples/causal_lm.py --config qwen3.local.toml --check-config
python examples/run_model.py --config qwen3.local.toml --check-config
```

两项配置检查都必须以零状态码结束。运行时资产路径和凭据只能放在 shell 环境中，
不能写入该 TOML。

## 8. 运行 Qwen3-0.6B 推理

```bash
python examples/run_model.py --config qwen3.local.toml
```

命令应输出生成结果并以零状态码结束。fused CausalLM 路径中，每个进程只允许一个
存活的 RPU resident 模型或 policy。

## 常见配置错误

| 现象 | 处理方式 |
|---|---|
| 外层或内层 checksum 失败 | 停止操作，重新取得匹配的归档与 checksum；不要混用 runtime set |
| 交付包 metadata 不是 v1.0.0 或状态不是 active | 停止操作，向分发方获取兼容 RhinoForge v1.0.0 release 的 active 交付包 |
| CMake 找不到 `rhino_launch` | 重新把上面派生的 Launch 前缀导出到 `CMAKE_PREFIX_PATH` |
| 无法加载 Launch 动态库 | 重新把其 `lib` 目录导出到 `LD_LIBRARY_PATH` |
| `RPU_KERNEL_LIB_PATH` 或相邻 manifest 不可读 | 从同一交付包重新安装 `RELEASE.txt` 指定的两个未改动文件 |
| `pip check` 或版本检查报告缺少 Accelerate | 不带 `--no-deps` 重新执行普通项目安装 |
| `torch.rpu` 不可用 | 请板卡管理员确认板卡配置和设备权限 |
| 模型 alias 无法解析 | 保持 `RPU_MODEL_CACHE/Qwen3-0.6B` 完整，并保留 TOML alias |
| 模型预检查拒绝该 profile | 使用精确 checkpoint revision 和仓库内 v1.0.0 TOML |

其他支持模型和控制项见[模型支持](model_support.zh.md)、[模型资产](model_assets.zh.md)、
[运行时配置](runtime_config.zh.md)和[模型执行与性能采集](model_testing.zh.md)。
