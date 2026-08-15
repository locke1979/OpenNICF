# Issue #86 Phase 0 — experiment manifest specification

The manifest is the reproducibility and safety boundary for every later arm.
It is sanitized JSON/YAML, version-controlled, and must not contain secrets,
production data, prompts copied from private evidence, or credentials.

```yaml
manifest_version: 1
state: DRAFT # DRAFT or EXECUTABLE; only EXECUTABLE may run
evaluation_id: issue86-architecture-v1
issue: 86
baseline_issue: 85
production: false
branch: issue/86-architecture-evaluation
corpus:
  text_manifest: path/to/reviewed-85-corpus-manifest.json
  multimodal_manifest: path/to/multimodal-corpus-v1.json
  query_manifest: path/to/reviewed-query-labels-v1.json
  chunking_contract_sha256: recorded digest
  filters:
    acl_scopes: [internal]
    domains: [sanitized-evaluation]
  top_k: [1, 3, 5, 10, 20, 30, 50]
  similarity: cosine
representation_types: [text, page_image, region_image, mixed]
spaces:
  text:
    id: TEXT_QWEN3_4B_Q4_K_M_768_V1
    model: Qwen/Qwen3-Embedding-4B
    model_revision: llama.cpp-pinned-model-revision
    quantization: Q4_K_M
    artifact_sha256: 2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b
    dimension: 768
    native_dimension: 2560
    projection: MRL prefix [0:768]
    normalized: true
    runtime: llama.cpp:a4a4c51f3d40e086b59b73b631b5c43c8fbf4504
    preprocessing: qwen3-query-instruction-v1
    modality: text
  vl:
    id: VL_QWEN3_2B_W4_768_V1
    model: exact model identity recorded before execution
    model_revision: exact revision recorded before execution
    quantization: W4
    artifact_sha256: recorded before execution
    dimension: 768
    normalized: true
    native_dimension: exact native dimension recorded before execution
    runtime: exact runtime recorded before execution
    preprocessing: exact preprocessing recorded before execution
    modality: multimodal
rerankers:
  text:
    id: TEXT_RERANKER_V1
    model: exact model identity recorded before execution
    revision: recorded before execution
    quantization: recorded before execution
    capabilities: [text]
    failure_policy: FAIL_QUERY
  multimodal:
    id: VL_RERANKER_V1
    model: exact model identity recorded before execution
    revision: recorded before execution
    quantization: recorded before execution
    capabilities: [text, page_image, region_image, mixed]
    failure_policy: FAIL_QUERY
runtime:
  gpu: NVIDIA GeForce GTX 1060 3GB
  cuda: 11.8.89
  llama_cpp_sha256: a4a4c51f3d40e086b59b73b631b5c43c8fbf4504
  device_target: sm_61
  seed: 86
fusion:
  method: rrf
  k_rrf: 60
  raw_score_fusion: false
arms:
  - id: T0
    spaces: [TEXT_QWEN3_4B_Q4_K_M_768_V1]
    reranker: null
    candidate_depths: [10, 20, 30, 50]
  - id: T1
    spaces: [TEXT_QWEN3_4B_Q4_K_M_768_V1]
    reranker: TEXT_RERANKER_V1
    candidate_depths: [10, 20, 30, 50]
  - id: T2
    spaces: [TEXT_QWEN3_4B_Q4_K_M_768_V1]
    reranker: TEXT_RERANKER_V1
    candidate_depths: [10, 20, 30, 50]
limits:
  reranker_candidate_depth: [10, 20, 30, 50]
  max_tokens: recorded per arm
  max_images: recorded per arm
  max_pixels: recorded per arm
  timeout_ms: recorded per arm
  batch_size: recorded per arm
decision:
  candidate_recall_before_rerank: required
  paired_bootstrap_iterations: 2000
  confidence: 0.95
  production_migration_authorized: false
```

Every result manifest must include the manifest digest, arm, model/runtime
hashes, source/reviewed-label digests, candidate and post-rerank metrics,
context-selection metrics, failure counts, resource telemetry, and the exact
feature flags. Raw scores from different spaces are retained as provenance but
are not numerically fused.

## RRF, deduplication, and failure semantics

For each candidate ID, RRF is exactly:

```text
RRF(candidate) = sum(1 / (k_rrf + rank_i))
```

`k_rrf` is a required positive manifest integer. Missing candidates contribute
zero. Ties in an input lane are resolved by stable candidate ID before ranks
are assigned. Repeated representation IDs in one lane count once at their
best rank. The fused order is tied by fused score, best original rank, then
candidate ID.

The required ordering is:

```text
ACL/domain/system filtering
-> independent ranked lists
-> RRF
-> candidate-recall measurement
-> modality-aware reranking
-> representation/evidence deduplication
-> bounded context selection
```

Deduplication is intentionally late. A text chunk and its page image share
provenance but remain separate candidates until after candidate recall and
reranking. Final selection limits candidates per source/page and records every
suppressed representation. Neighbor/page crowding limits are manifest values;
they are not hidden score adjustments.

Reranker failure policy is explicit per reranker:

- `FAIL_QUERY`: return no reranked result and record the failure;
- `FALLBACK_TO_FUSED_ORDER`: retain pre-rerank RRF order, set
  `reranker_applied=false`, and record the fallback; it is not a successful
  reranker result;
- `SKIP_UNSUPPORTED_CANDIDATE`: only modality-aware routing may use this for a
  candidate assigned to another supported lane; it must record the candidate
  and reason and may not silently discard relevant evidence.

Every result records requested and actual reranker, eligible/scored/skipped
counts, failure classification, timeout/OOM/malformed-output state, fallback
policy, and whether fallback was invoked. Primary reranker comparisons exclude
fallback results or report them as a separate stratum.
