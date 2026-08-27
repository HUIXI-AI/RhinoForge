# rhinoforge-port

[简体中文](README.zh.md) | English

Portable AI-agent workflow for assessing or porting one Hugging Face inference
profile to RhinoForge. It uses the vendor-neutral `.agents/skills` layout and
does not require an external plugin.

## Use

Ask an AGENTS.md-aware agent to use `rhinoforge-port` in one mode:

- **Assess:** review feasibility and return one of Existing path, Adapter-only,
  Runtime extension, New fused subsystem, or Blocked. It does not edit code.
- **Port:** implement an accepted, certified, non-Blocked assessment and run
  the applicable board-free and authorized hardware gates.

Examples:

```text
Use rhinoforge-port to assess Qwen/Qwen3-0.6B for FP16 prefill and decode.
Use rhinoforge-port to port the accepted profile from the assessment report.
```

The complete workflow is in [SKILL.md](SKILL.md). Family-specific checklists,
the public capability map, and the portable assessment report template are
under [references](references/).

## Contents

- [Assessment evidence rules](references/assessment.md)
- [Public capability map](references/capability-map.md)
- [Assessment report template](references/report-template.md)
- Family playbooks: [CausalLM](references/families/causal-lm.md),
  [Vision](references/families/vision.md), [VLM](references/families/vlm.md),
  and [VLA](references/families/vla.md)

The skill is intentionally tool-neutral. It contains no `.claude` directory,
generated orchestration workflow, autonomous self-modification loop, internal
benchmark corpus, or privileged hardware command. An agent runtime may use its
own isolated reviewers, task list, and command runner while preserving the
same evidence and stop rules.
