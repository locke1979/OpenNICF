# Issue #86 Phase 0 — focused test plan

These are the implementation gates before model execution and before any
evaluation code can be enabled. They are intentionally testable without model
downloads.

## Contract and isolation

- text and VL 768-D vectors reject cross-space comparison;
- representation IDs are stable for identical canonical inputs;
- changed rendition bytes or renderer revision creates a new identity;
- text and VL vectors cannot be written under one representation-space key;
- production defaults remain disabled and active production space is unchanged;
- sanitized manifests reject credentials, absolute production paths, and raw
  evidence payloads.

## Provenance and policy

- every representation inherits source/version/artifact, ACL, domain, system,
  component, environment, and evidence metadata;
- filtering occurs before dense retrieval, RRF, and reranker dispatch;
- reranking preserves candidate IDs and provenance exactly;
- QwenAgent receives only allow-listed provenance-aware tools;
- visual/document content cannot change workflow, limits, tools, or routing;
- image-only candidates are rejected by a text-only reranker rather than
  silently projected.

## Ingestion and idempotence

- text bundle commits when visual rendering fails;
- visual failure is retryable or quarantined with a bounded classification;
- unchanged rendition hash avoids duplicate object/DB work;
- re-ingestion is idempotent;
- page resolution, pixel count, image count, archive expansion, and batch size
  limits fail closed;
- malformed visual input cannot corrupt immutable originals or text chunks.

## Retrieval, reranking, and context

- RRF is deterministic under ties and never combines raw scores;
- candidate recall is recorded before reranking;
- reranker timeout/OOM/malformed output follows explicit fail-closed policy;
- source/page/region deduplication prevents representation crowding;
- context selection runs after reranking and enforces token/image/pixel/source
  budgets;
- retrieval events and result manifests identify first-stage, reranked, and
  context-retained phases.

## Planned execution order

1. Add pure contract/unit tests and sanitized manifest validation.
2. Add in-memory representation store and deterministic RRF tests.
3. Add bounded renderer/embedding failure tests using synthetic sanitized
   fixtures.
4. Run existing focused suite and complete suite.
5. Only after Phase 0 review, acquire models and run evaluation arms.
