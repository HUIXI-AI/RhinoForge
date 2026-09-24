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
