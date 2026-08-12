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
object_key: nullable for text backed directly by an existing chunk; required for new renditions
mime_type: rendition MIME type
pixel_width: nullable bounded integer
pixel_height: nullable bounded integer
render_name: renderer/parser name
render_version: renderer/parser revision
rendition_hash: SHA-256 of canonical rendition bytes/manifest
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

Embedding model, quantization, dimensions, normalization, and
`embedding_space_id` belong to the separate embedding-variant record, not to
the rendition identity.

`representation_id` is stable over the canonical tuple
`source_version_id`, `representation_type`, stable locator, optional
`parent_chunk_id`, `rendition_hash`, `render_name`, and `render_version`.
`embedding_space_id` is deliberately excluded. A changed renderer revision or
rendition bytes creates a new representation identity; reprocessing an
unchanged rendition is an idempotent no-op. BF16/W4/W8 and future revisions
attach to the same representation through separate embedding rows.

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
  object_key TEXT,
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
  attempt_count INTEGER NOT NULL DEFAULT 0,
  first_attempt_at TIMESTAMPTZ,
  last_attempt_at TIMESTAMPTZ,
  last_error_at TIMESTAMPTZ,
  last_error TEXT,
  UNIQUE(source_version_id, representation_type, page_locator, parent_chunk_id, rendition_hash, render_name, render_version)
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

### Integrity rules

- `component_id` follows the current OpenNICF contract: it is required as a
  non-null normalized identifier; use `unknown-component` when no finer value
  is available, rather than using SQL `NULL`.
- `object_key` is nullable only for a text representation that is exactly
  backed by an existing immutable `knowledge_chunks` row. Image, region, and
  mixed renditions must have an object-store key. The repository method must
  enforce this by representation type.
- A representation write validates that source, source version, and artifact
  refer to the same existing chain. Durable SQL should use composite foreign
  keys (or a validated repository transaction) so an artifact cannot be
  attached to another source version.
- `parent_chunk_id`, when present, must belong to the same `source_version_id`.
  The isolated implementation rejects mismatches before insertion.
- Only `ready` representations with a compatible embedding variant are
  eligible for retrieval. `pending`, `failed`, `quarantined`, and `retired`
  rows remain auditable but are excluded by the candidate query.
- Page/region locators are stable canonical strings. Uniqueness includes
  source version, representation type, parent chunk, locator, rendition hash,
  and renderer identity so separate regions/pages cannot collide.
- Retirement is a lifecycle update that preserves the immutable original and
  all provenance. It does not cascade-delete source, artifact, chunk, or
  embedding history.

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
