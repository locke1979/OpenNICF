# Issue #86 Phase 0 — schema and API compatibility

## Existing persistence contract

The current PostgreSQL contract is anchored by:

- `knowledge_sources`;
- `knowledge_source_versions`;
- `knowledge_artifacts`;
- `knowledge_chunks`;
- `knowledge_embeddings`;
- `knowledge_embedding_spaces` and `knowledge_embedding_migrations`;
- `knowledge_retrieval_events`;
- namespace and code-index tables.

`knowledge_embeddings` is already space-aware through
`(chunk_id, embedding_space_id)`. Equal vector dimensions do not imply
compatibility.

## Proposed representation contract

The evaluation requires one new logical record for every derived text, page,
region, or mixed representation:

```yaml
representation_id: stable content-addressed identity
representation_type: text | page_image | region_image | mixed
source_id: existing source identity
source_version_id: existing immutable version identity
artifact_hash: existing immutable artifact hash
parent_chunk_id: nullable existing text chunk identity
page_locator: nullable stable page/region locator
object_key: rendition object-store key
mime_type: rendition MIME type
pixel_width: nullable bounded integer
pixel_height: nullable bounded integer
render_name: renderer/parser name
render_version: renderer/parser revision
rendition_hash: SHA-256 of canonical rendition bytes/manifest
embedding_space_id: explicit independent semantic space
model: model identity
model_revision: model revision or commit
quantization: none | bf16 | fp16 | w4 | w8 | q4_k_m | other
dimensions: positive integer
normalized: boolean
lifecycle_status: pending | ready | failed | quarantined | retired
failure_classification: nullable bounded enum
acl_scope: inherited immutable ACL
domain_id: inherited domain
system_id: inherited system
component_id: inherited component
environment: inherited environment
evidence_type: inherited evidence class
metadata: sanitized bounded JSON metadata
```

`representation_id` is stable over a canonical tuple of source version,
parent locator, representation type, rendition hash, and embedding space. A
changed renderer revision or bytes creates a new representation/rendition
identity; reprocessing unchanged bytes is an idempotent no-op.

## Proposed database shape (evaluation design only)

Do not apply this migration in production during issue #86.

```sql
knowledge_representations (
  representation_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES knowledge_sources(source_id),
  source_version_id TEXT NOT NULL REFERENCES knowledge_source_versions(source_version_id),
  artifact_hash TEXT NOT NULL REFERENCES knowledge_artifacts(artifact_hash),
  parent_chunk_id TEXT REFERENCES knowledge_chunks(chunk_id),
  representation_type TEXT NOT NULL,
  page_locator TEXT,
  object_key TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  pixel_width INTEGER,
  pixel_height INTEGER,
  render_name TEXT NOT NULL,
  render_version TEXT NOT NULL,
  rendition_hash TEXT NOT NULL,
  lifecycle_status TEXT NOT NULL,
  failure_classification TEXT,
  acl_scope TEXT NOT NULL,
  domain_id TEXT NOT NULL,
  system_id TEXT NOT NULL,
  component_id TEXT NOT NULL,
  environment TEXT NOT NULL,
  evidence_type TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(source_version_id, representation_type, page_locator, rendition_hash)
)

knowledge_representation_embeddings (
  representation_id TEXT NOT NULL REFERENCES knowledge_representations(representation_id),
  embedding_space_id TEXT NOT NULL REFERENCES knowledge_embedding_spaces(embedding_space_id),
  model TEXT NOT NULL,
  model_revision TEXT NOT NULL,
  quantization TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  normalized BOOLEAN NOT NULL,
  vector vector NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY(representation_id, embedding_space_id)
)
```

Dimension-specific ANN materialization must be created only after a target
space is validated. Legacy vectors and active production indexes are not
cast, rebuilt, or mixed by this evaluation.

## Provider-neutral APIs

```python
class Reranker(Protocol):
    @property
    def info(self) -> RerankerInfo: ...

    def score(
        self,
        query: RerankQuery,
        candidates: Sequence[RerankCandidate],
        *,
        limits: RerankLimits,
    ) -> RerankResult: ...
```

The contract must validate capability before dispatch:

- text reranker accepts text candidates only;
- multimodal reranker accepts declared image/mixed inputs;
- each candidate preserves `representation_id`, `chunk_id`, source/version,
  artifact hash, locator, ACL/domain/system metadata;
- limits include candidate pairs, tokens, images, pixels, batch size, timeout;
- result includes model/revision/quantization, route, elapsed time, and failure;
- ties are resolved by original rank then stable candidate ID;
- timeout, OOM, unsupported modality, and malformed output are fail-closed or
  use an explicitly configured evaluation fallback recorded in the manifest.

## Compatibility guarantees

1. Existing `ChunkRecord`, `EmbeddingRecord`, `EvidenceHit`, `KnowledgeStore`,
   and QwenAgent tool signatures remain source-compatible through optional
   fields or adapter wrappers.
2. Existing `embedding_space_id` checks remain authoritative. Text and VL
   spaces have different IDs even at 768 dimensions.
3. ACL/domain/system filtering happens before candidate retrieval and before
   reranker dispatch.
4. Existing provenance fields are copied, never regenerated from untrusted
   representation content.
5. Text ingestion and text retrieval remain usable when visual work fails.
6. `KnowledgePlatform.search` and current production feature defaults retain
   current behavior when the evaluation flag is absent.
7. Evaluation result manifests are append-only, sanitized, and independent of
   `knowledge_retrieval_events` in production.
