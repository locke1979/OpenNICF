# Retrieval benchmark (Phase 2, issue #52)

The reusable workload lives in [`tests/fixtures/retrieval_benchmark/v1`](../tests/fixtures/retrieval_benchmark/v1). `manifest.json` is versioned and contains the SHA-256 digest, source identity, domain, ACL, parser version, and language for every sanitized fixture. It covers Portuguese and English technical documentation, Java, SQL, PowerShell, application logs, Splunk-style CSV, a schema, integration documentation, and query output.

Run the deterministic local benchmark with:

```bash
PYTHONPATH=src python -m opennicf.knowledge.benchmark \
  --manifest tests/fixtures/retrieval_benchmark/v1/manifest.json \
  --output retrieval-benchmark.json
```

The JSON result is a provenance-backed result manifest. Every ranked hit carries `source_id`, `source_version_id`, `artifact_hash`, `chunk_id`, `locator`, `excerpt_hash`, and `embedding_space_id`; the report also records the fixture-manifest digest and execution path. It does not copy fixture text into the report.

## Methods and metrics

The workload compares lexical-only, vector-only, the existing hybrid weights, and symbol-aware code retrieval. Symbol-aware evaluation is intentionally limited to the Java, SQL, and PowerShell queries. All methods use the same normalized domain/ACL filters and deterministic chunk tie-breakers.

- `Recall@5`: fraction of queries with the expected source in the first five chunks.
- `MRR`: reciprocal rank of the first expected source, averaged over queries.
- `latency_ms`: measured per-query retrieval duration; ranking and quality metrics are deterministic, while latency is host-dependent.
- `returned_chunks`: number of ranked chunks before context compaction.
- `context_tokens_after_compaction`: Unicode word-token count after duplicate removal and the bounded four-chunk/256-token context budget.

The baseline captured on 2026-08-10 with the dependency-free 768-dimensional hashing fallback was:

| mode | queries | Recall@5 | MRR | median latency (ms) | mean context tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical | 10 | 0.80 | 0.800 | 1.0 | 72.0 |
| vector | 10 | 0.80 | 0.733 | 5.6 | 70.5 |
| hybrid | 10 | 0.80 | 0.733 | 7.1 | 73.1 |
| symbol-aware | 3 | 1.00 | 1.000 | 0.6 | 25.3 |

The local fallback is explicit: the result manifest records `knowledge_path: KnowledgePlatform.in_memory`, `embedding_provider: LOCAL`, `fallback: true`, and `fallback_reason: deterministic_cpu_backend`. A production run should retain this manifest and replace only the platform factory with the deployed PostgreSQL/object-store path; it must not compare vectors from a different `embedding_space_id`.

## Tuning decision

The measured baseline exposed an identifier-tokenization miss: queries such as `artifact hash source version` could not match stored `artifact_hash` and `source_version_id` fields. The narrow tokenizer change preserves whole identifiers and adds their components. On the same corpus it raised lexical Recall@5/MRR from `0.80/0.800` to `1.00/0.950` and hybrid Recall@5/MRR from `0.80/0.733` to `1.00/0.875`; vector scores were unchanged. No semantic weight was changed because the only measured embedding is the deterministic hash fallback, not the deployed Qwen/Gemini distribution. The benchmark and isolation tests are the gate for a later measured weight change.

The focused tests also verify that a public or foreign-domain decoy is excluded, a mismatched embedding space fails closed, cloud embeddings are rejected before transport for `local_only`, and the issue #52 machine-readable contract remains valid.
