# Model validation

[简体中文](validation_policy.zh.md) | English

Validation should answer the question raised by the change. Use the smallest
meaningful checks and reuse applicable existing results.

- Run affected host tests once. Passing focused checks do not require a full suite.
- For a changed RPU execution path, use one representative smoke test. Replay
  checks normally use 3–10 repetitions. Performance checks use 2 warmups and
  5–10 measured iterations, with the same checkpoint, precision and workload.
- Check the relevant failure boundary before weight transformation or KV mutation,
  and verify input freshness, retained outputs or BUILD/REPLAY ownership when the
  change affects them. Do not add unrelated lifecycle or numerical checks.
- Investigate wrong shapes, invalid values, stale inputs, incorrect routes or cache
  corruption as correctness issues. After confirming the computational path,
  record ordinary accumulated numerical differences and quantization differences;
  no universal cosine threshold or exact generated-token match is required.
- Stop after the requested behavior and directly affected regression checks pass.
  A failure may justify a small diagnostic follow-up tied to that failure.

Full model/precision/length matrices, more than 30 replay or benchmark iterations,
repeated clean-room runs, cryptographic evidence closures, custom publication
harnesses and layered certification gates require an explicit request. Ordinary
fixes do not start a model-port certification campaign.

Check that the board is idle and acquire its existing lock before hardware work.
Use an isolated environment bound to the intended worktree and runtime assets.
Report the tested scope and remaining limitations without transferring results
between models or builds. See [performance measurement](performance.md).

## Numerical evaluation

Use the three layers below for numerical evaluation.
Inherited floating-point cutoffs such as `MSE <= 1e-4`, `max_abs <= 0.05`,
their row-wise variants, and fixed cosine floors are not model acceptance
gates. This supersedes older instructions to preserve those profile-local
gates. Do not rename them as calibrated/composite gates or replace them with
another fixed error ceiling. Keep historical measurements and verdicts as
history; do not silently rewrite them as current passes.

1. **Runtime correctness remains mandatory.** Check the declared shape and
   dtype, finite outputs, input freshness, correct execution routes, and
   KV/cache/SPM/DMA and Graph lifecycle invariants. Artifact identity,
   integer/structural equality and explicitly deterministic replay are distinct
   contracts; this correction does not relax them.
2. **Compare implementations under matching semantics.** Bind the checkpoint,
   quantization recipe, inputs, preprocessing, reference arithmetic, and output
   space. For quantized execution, first use the same stored quantized weights
   and scale semantics; comparison with the original floating-point model
   additionally measures quantization effects. Record per-row/position error
   distributions and useful magnitude metrics (for example MSE, MAE, relative
   L2 or max-abs), not just flattened cosine. For LLMs, compare aligned token
   prefixes, top-k/logprobs and near-tie behavior. Exact generated text alone
   neither establishes nor refutes general numerical correctness. Diagnose
   suspicious differences against the actual route, precision, inputs and
   reference; ordinary finite rounding/accumulation differences do not create
   an automatic failure from one of the retired cutoffs.
3. **Evaluate model/task quality separately.** Use a stated dataset and
   protocol with the applicable task metric, such as LLM accuracy/NLL/PPL or
   VLA denormalized actions, trajectories and closed-loop success. Preserve
   independently specified task-quality requirements. A short output comparison
   is not a substitute for task evaluation, and missing task evidence stays
   unmeasured/uncertified. Removing a scalar gate does not certify a model.

Report runtime results, numerical observations, and task-quality results
separately, including untested scope. Same-runtime batch/chunk/fused regression
checks remain useful, but a historical scalar threshold does not become a
current acceptance rule merely because both paths run on the same device.
Operator unit tests against a specified mathematical reference are distinct;
their tolerances are not transferable end-to-end model-quality budgets.

These methods follow the distinction visible in upstream tools, not a shared
upstream numerical threshold:

- [TensorRT accuracy guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
  discusses precision-dependent rounding, overflow and sensitive operations;
  [Polygraphy comparisons](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/polygraphy/tool/args/comparator/compare.html)
  configure tolerances and error statistics per output tensor.
- [vLLM model tests](https://docs.vllm.ai/en/latest/contributing/model/tests/)
  distinguish exact generated output from top-k logprob similarity against
  Hugging Face; these are test choices, not one cross-model error budget.
- [TensorRT-LLM evaluation](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-eval.html)
  uses task metrics including accuracy and ROUGE. Its development evaluator
  does not by itself establish production or robot-task acceptance.
