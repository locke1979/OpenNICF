# Issue #90 bounded dependency graph (proposal only)

All proposed children have parents #90 and #86, `subagent-pipeline: autorun`, evaluation-only scope, and no deploy/merge authorization.

1. **VL artifact/runtime/licence qualification** — depends on #88. Accept when exact revisions, licenses, local hashes, templates, pixel bounds, and a Pascal-compatible or explicitly alternative isolated runtime pass validation.
2. **Sanitized multimodal corpus and human review** — depends on model-free corpus tooling. Accept at >=100 authorized representations and >=50 genuinely reviewed queries with complete hard negatives and immutable digests.
3. **Multimodal ingestion/rendering implementation** — depends on representation contracts. Accept when lifecycle, provenance, policy propagation, idempotence, quarantine/retry and resource-bound tests pass.
4. **VL embedding/reranker benchmark** — depends on children 1-3 and an `EXECUTABLE` manifest. Accept only with measured paired metrics and telemetry; no production routes.
5. **Topology/concurrency evaluation** — depends on the benchmark runtime. Accept with measured concurrency 1/2/4, restart/reload, OOM/failure, storage and queue results.

Dependency graph: `#88 -> {artifact qualification, corpus/review, rendering} -> benchmark -> topology`. Issue creation is intentionally left to the pipeline coordinator to avoid duplicates.
