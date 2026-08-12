# Issue #86 Phase 0 — blockers and dependencies

## Dependencies

- #85 decision-grade reviewed text corpus and accepted labels; #86 must not
  contaminate #85 base-embedding metrics.
- Existing Qwen 4B text artifact/runtime from #85:
  artifact SHA `2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b`,
  llama.cpp SHA `a4a4c51f3d40e086b59b73b631b5c43c8fbf4504`, CUDA 11.8, sm_61.
- An authoritative, license-compatible Qwen VL-2B embedding artifact and a
  bounded multimodal reranker must be identified before downloads.
- A fixed generator/evaluator contract is required for RAG comparisons.

## Findings requiring review

1. Current production compaction is clearly implemented in
   `knowledge/benchmark.py:compact_hits`, but not clearly enforced at the
   QwenAgent tool boundary. The evaluation must add a post-rerank boundary
   without silently changing production behavior.
2. Existing `OpenNICFTools.search_evidence` is a reduced compatibility surface;
   it omits some full artifact/version/excerpt/model metadata. The experiment
   must use the richer domain package or extend the adapter compatibly.
3. Existing PostgreSQL schema has vector storage and space metadata, but no
   representation table and no evidenced database-native ANN/full-text index
   in the reviewed migrations. Evaluation should remain isolated and not
   assume production ANN changes.
4. Parser and ingestion failures include generic exceptions. Visual failures
   need a representation-scoped retry/quarantine classification rather than a
   broad job failure that loses text availability.
5. GTX 1060 3 GB is unlikely to hold all embedding and reranker models
   simultaneously. Dedicated workers, CPU reranking, conditional VL routing,
   and serialized residency must be measured as separate topologies.

## Explicit non-blockers

- No open PR currently advertises an implementation of issue #86.
- Existing issue #85 remains independent and open.
- GitHub authentication is available for issue/PR updates; it is not a model
  execution dependency.

## Safety status

```text
PRODUCTION = UNCHANGED
ARCHITECTURE_EVALUATION = PHASE_0_DESIGN_READY_FOR_REVIEW
RECOMMENDATION = NEEDS_EVIDENCE
MODEL_DOWNLOADS = 0
DEPLOYMENTS = 0
```
