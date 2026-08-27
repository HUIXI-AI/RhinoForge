# rhinoforge-port

简体中文 | [English](README.md)

这是一个可移植的 AI agent 工作流，用于评估或移植一个 Hugging Face 推理配置到
RhinoForge。它采用通用的 `.agents/skills` 目录，不依赖特定插件或 agent 产品。

## 使用方式

请 AGENTS.md-aware agent 以一种模式使用 `rhinoforge-port`：

- **Assess：** 只读检查可行性，给出 `Existing path`、`Adapter-only`、
  `Runtime extension`、`New fused subsystem` 或 `Blocked`，不修改代码。
- **Port：** 从已经接受、`certified` 且非 `Blocked` 的评估开始，完成实现并运行
  适用的无板卡门禁，以及明确授权后的硬件门禁。

示例：

```text
使用 rhinoforge-port 评估 Qwen/Qwen3-0.6B 的 FP16 prefill 与 decode。
使用 rhinoforge-port 按已审核的评估报告移植该配置。
```

完整工作流见 [SKILL.md](SKILL.md)。

## 内容导航

- [评估证据规则](references/assessment.md)
- [公开能力地图](references/capability-map.md)
- [评估报告模板](references/report-template.md)
- 模型类型 playbook：[CausalLM](references/families/causal-lm.md)、
  [Vision](references/families/vision.md)、[VLM](references/families/vlm.md) 和
  [VLA](references/families/vla.md)

这个 skill 刻意保持工具无关：不包含 `.claude` 目录、生成式编排 workflow、自动
自修改回路、内部 benchmark 语料或特权硬件命令。不同 agent runtime 可以使用自己的
隔离 reviewer、任务列表和命令执行器，但必须保持相同的证据规则与停止条件。
