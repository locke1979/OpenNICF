# Issue #86 Phase 0 — experiment manifest specification

The manifest is the reproducibility and safety boundary for every later arm.
It is sanitized JSON/YAML, version-controlled, and must not contain secrets,
production data, prompts copied from private evidence, or credentials.

```yaml
manifest_version: 1
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
spaces:
  text:
    id: TEXT_4B_Q4_768
    model: Qwen/Qwen3-Embedding-4B
    quantization: Q4_K_M
    artifact_sha256: 2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b
    dimension: 768
    native_dimension: 2560
    projection: MRL prefix [0:768]
    normalized: true
  vl:
    id: VL_2B_W4_768
    model: exact model identity recorded before execution
    quantization: W4
    artifact_sha256: recorded before execution
    dimension: 768
    normalized: true
rerankers:
  text:
    model: exact model identity recorded before execution
    revision: recorded before execution
    quantization: recorded before execution
    capability: [text]
  multimodal:
    model: exact model identity recorded before execution
    revision: recorded before execution
    quantization: recorded before execution
    capability: [text, page_image, region_image, mixed]
runtime:
  gpu: NVIDIA GeForce GTX 1060 3GB
  cuda: 11.8.89
  llama_cpp_sha256: a4a4c51f3d40e086b59b73b631b5c43c8fbf4504
  device_target: sm_61
  seed: 86
arms:
  T0: lexical + symbol + text dense, no reranker
  T1: text dense -> text reranker
  T2: lexical + symbol + text dense -> RRF -> text reranker
  M0: VL dense, no reranker
  M1: VL dense -> VL reranker
  M2: T2 + VL -> RRF -> text reranker for eligible text candidates
  M3: T2 + VL -> RRF -> modality-aware reranking
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
