# G0.5 适配接口

RhinoForge 提供 `patch_g05_policy_for_rpu` 及 Vision、Language、Action 组件适配入口，见 [G0.5 adapter](../../../python/rpu_backend/adapters/g05/__init__.py)。模型源码、checkpoint 和预处理由调用方提供，并须满足所选适配路径的安装检查。

## 组件与输入

Vision 处理相机图像，Language 处理文本与视觉前缀，Action 使用前缀状态生成动作。调用方负责按模型定义准备相机顺序、图像、文本、状态和动作输入；组件之间的张量布局和缓存归属见[组件交接](../../../knowledge/concepts/component-handoff.md)。

精度和执行参数在安装前绑定。FP16、W8A16 与动作 NVFP4 使用各自的权重转换路径；某个组件的量化支持不能推导为任意 checkpoint 或完整策略的支持。安装检查、输入约束和实际验证结果共同决定适用范围。

## 集成与资产

先准备相互匹配的模型源码、processor、checkpoint 和输入，再由适配器检查模型结构与执行参数。每个实例只转换一次权重；需要改变安装参数时重新加载干净实例。重复请求遵守[Graph 与运行时生命周期](../../../docs/architecture.md)。

公开模型项目和许可见[第三方声明](../../../THIRD_PARTY_NOTICES.md)。模型资产由调用方取得；RhinoForge 不分发这些权重。
