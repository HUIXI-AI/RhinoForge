# 入门指南

简体中文 | [English](getting_started.md)

本指南在独立 Python 环境安装当前 RhinoForge 源码，并运行公开的 Qwen3-0.6B 示例。
模型和运行时资产分别获取。[示例目录](../examples/README.md)说明配置和用法，
模板不构成模型质量认证。

## 1. 准备运行时和独立环境

需要 Python 3.12、C++17 编译器和已配置的板卡 SDK/runtime。按
[运行时资产](runtime_assets.zh.md)获取匹配的外部资产，并设置
`RHINO_LAUNCH_ROOT`、`RPU_KERNEL_LIB_PATH` 及动态库路径。

为当前 checkout 创建新环境，与其他 RhinoForge 或 `rpu_backend` 安装隔离。
在同一个 shell 中运行：

```bash
export RHINOFORGE_SOURCE="/absolute/path/to/RhinoForge"
export RHINOFORGE_ENV="/absolute/path/to/new/rhinoforge-env"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3.12 -m venv "$RHINOFORGE_ENV"
. "$RHINOFORGE_ENV/bin/activate"
cd "$RHINOFORGE_SOURCE"

python -m pip install "torch==2.10.0" "scikit-build-core>=0.12,<0.13" "cmake==4.1.3" "ninja==1.13.0"
python -m pip install . --no-build-isolation \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
python -m pip check
python examples/verify_install.py --check-config
```

`--check-config` 只检查安装元数据和模块位置，不导入 Torch 或 RhinoForge，
不加载原生运行时，也不访问 RPU；它不验证原生库兼容性或设备是否可用。

项目依赖以 [pyproject.toml](../pyproject.toml) 为准。视觉模型安装命令中的 `.` 改为
`'.[vision]'`，策略模型改为 `'.[vla]'`。可编辑安装和 wheel 构建参考
[源码构建指南](development_build.md)。

## 2. 准备公开模型

将公开 checkpoint 下载到示例别名使用的目录：

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
mkdir -p "$RPU_MODEL_CACHE"
hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-0.6B"
cp examples/configs/qwen3/text/0_6b/fp16.toml qwen3.local.toml
python examples/run_model.py --config qwen3.local.toml --check-config
```

已有同一 checkpoint 的完整副本时，可直接放入该目录。模型许可和访问条款与
RhinoForge 源码许可分别适用。所有示例配置均在
[`examples/configs`](../examples/README.md) 下按模型归档。

## 3. 在空闲 RPU 上运行

运行前先检查本机 RPU 是否空闲，并取得平台设备锁。共享机器需与其他使用者协调。
然后执行：

```bash
python examples/run_model.py --config qwen3.local.toml
```

runner 输出生成结果和计时。需要诊断基础设备配置时，可用
`python examples/verify_install.py` 做一次小规模 host/device 拷贝；模型已运行成功时
无需额外执行。复用本次成功结果，不展开模型矩阵。

若导入了其他 checkout，激活目标环境并查看
`python examples/verify_install.py --check-config` 输出的模块位置。模型别名无法
解析时，检查 `RPU_MODEL_CACHE/Qwen3-0.6B`；运行时资产问题见
[运行时资产](runtime_assets.zh.md)。

后续参考[模型资产](model_assets.zh.md)、[运行时配置](runtime_config.zh.md)和
[模型运行与性能分析](model_testing.zh.md)。
