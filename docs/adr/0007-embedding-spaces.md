# ADR 0007: Provider-safe embedding spaces

Status: accepted

Issue: #43

OpenNICF uses one provider-neutral API, `embed(texts, purpose, dimension)`,
with 768 as the canonical normalized cosine-search default. Equal dimensions
do not imply equal semantic spaces: Qwen3-Embedding-0.6B,
`gemini-embedding-001`, and `gemini-embedding-2` each have an explicit
`embedding_space_id` containing model and revision. Every vector records its
space, provider, model revision, dimension, normalization, purpose, and
timestamp while pointing to the same immutable chunk/provenance record.

Qwen is the local-first adapter. It detects CUDA at runtime, uses FP16 where
viable, bounds input to RAG-sized chunks, batches conservatively, retries GPU
OOM with smaller batches, and falls back to an explicit CPU backend. Query
instructions and document behavior remain inside the adapter. Gemini adapters
use runtime-only credentials and enforce the same normalized 768-d contract.
`local_only` evidence fails closed before any Google request.

PostgreSQL stores `embedding_space_id` and registers spaces separately. The
legacy unbounded pgvector column is preserved without an unsafe in-place
`vector(768)` cast or HNSW index; its compatibility path uses a space index
and an explicit relation predicate. A validated deployment may create a
dimension-specific ANN table/index per registered space. Retrieval applies
ACL/domain/system filters, selects the explicit active space, then performs
hybrid lexical, symbol, and vector retrieval. Cross-space comparison raises
an error even when dimensions match.

Migration registers a target space, re-embeds chunks in resumable idempotent
batches, preserves provenance, builds and validates its ANN index, and only
then atomically switches the active selector. The old space remains available
until explicit retention retires it. Source migration checksums fail closed on
drift and existing vectors are never destructively rebuilt by this change.
