# ADR 0004: Canonical knowledge platform and evidence store

Status: accepted

Issue: #5

OpenNICF needs a durable evidence layer for audits, retrieval, and provenance-aware model context. The implementation must preserve immutable originals, deduplicate identical content, version modified sources, and keep ACL filtering ahead of model exposure.

We use PostgreSQL + pgvector for metadata, chunk, embedding, retrieval-event, and finding records. The schema and indexes are source-controlled as SQL migrations checked into the repository. This keeps the database shape reviewable and replayable while still allowing the existing PostgreSQL LXC 156 to remain the single authoritative database host.

Original files and generated artifacts are stored through an object-store abstraction. The default non-production backend is a filesystem object store keyed by content hash so that identical bytes resolve to the same immutable object. The abstraction leaves room for an S3-compatible backend later without changing the evidence model.

Embeddings are routed through a local-first adapter. If a GPU-capable backend is available it may be selected, but CPU fallback is always available through a deterministic model with explicit model and dimension metadata. Batch sizing is constrained by a memory budget heuristic so the same interface can run on low-VRAM devices.

Retrieval is hybrid: PostgreSQL filters and ACL checks narrow the candidate set, and the application then blends semantic and lexical scoring while preserving locators for citation. Retrieval events store the filters, route, selected chunks, and scores for auditability.

